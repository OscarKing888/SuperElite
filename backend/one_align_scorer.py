"""
SuperElite - One-Align scorer
Based on q-future/one-align (quality + aesthetics).
"""

import inspect
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image


class OneAlignScorer:
    """One-Align scorer: quality + aesthetics."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        quality_weight: float = 0.4,
        aesthetic_weight: float = 0.6,
    ):
        self.model_path = model_path or "q-future/one-align"
        self.quality_weight = quality_weight
        self.aesthetic_weight = aesthetic_weight

        self.model = None
        self.device = None

        print(f"[OneAlign] 模型: {self.model_path}")
        print(f"[OneAlign] 权重: Quality={quality_weight}, Aesthetic={aesthetic_weight}")

    def _select_device(self) -> str:
        """Select best available device (MPS first)."""
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _patch_transformers_compatibility():
        """
        transformers 5.x removed some legacy APIs used by One-Align remote code.
        """
        try:
            from transformers import pytorch_utils

            if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

                def find_pruneable_heads_and_indices(heads, n_heads, head_dim, already_pruned_heads):
                    mask = torch.ones(n_heads, head_dim)
                    heads = set(heads) - already_pruned_heads
                    for head in heads:
                        mask[head] = 0
                    mask = mask.view(-1).contiguous().eq(1)
                    index = torch.arange(len(mask))[mask].long()
                    return heads, index

                pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
                print("[OneAlign] 已修复 transformers 5.0 兼容性 (find_pruneable_heads_and_indices)")
        except Exception as e:
            print(f"[OneAlign] 警告: transformers 兼容性补丁失败 ({e})")

    @staticmethod
    def _patch_llama_rotary_embedding():
        """Patch RoPE-related APIs for One-Align on modern transformers."""
        try:
            import transformers.models.llama.modeling_llama as llama_modeling

            # 1) apply_rotary_pos_emb old signature compatibility
            if hasattr(llama_modeling, "apply_rotary_pos_emb"):
                original_apply = llama_modeling.apply_rotary_pos_emb
                apply_sig = inspect.signature(original_apply)
                if (
                    "position_ids" not in apply_sig.parameters
                    and not getattr(original_apply, "_one_align_compat", False)
                ):

                    def patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                        # Old callers pass `position_ids` as the 5th positional arg.
                        if isinstance(position_ids, int):
                            unsqueeze_dim = position_ids
                        return original_apply(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

                    patched_apply_rotary_pos_emb._one_align_compat = True
                    llama_modeling.apply_rotary_pos_emb = patched_apply_rotary_pos_emb

            # 2) LlamaRotaryEmbedding __init__/forward compatibility
            LlamaRotaryEmbedding = llama_modeling.LlamaRotaryEmbedding

            original_init = LlamaRotaryEmbedding.__init__
            init_sig = inspect.signature(original_init)
            if (
                "config" in init_sig.parameters
                and not getattr(original_init, "_one_align_compat", False)
            ):

                def patched_init(self, *args, **kwargs):
                    # Old API: LlamaRotaryEmbedding(dim, max_position_embeddings=..., base=...)
                    if args and isinstance(args[0], int):
                        dim = args[0]
                        max_pos = kwargs.pop("max_position_embeddings", 2048)
                        base = float(kwargs.pop("base", 10000.0))
                        device = kwargs.pop("device", None)
                        compat_cfg = SimpleNamespace(
                            max_position_embeddings=max_pos,
                            rope_parameters={"rope_type": "default", "rope_theta": base},
                            head_dim=dim,
                            hidden_size=dim,
                            num_attention_heads=1,
                        )
                        return original_init(self, compat_cfg, device=device)

                    cfg = args[0] if args else kwargs.get("config")
                    rope_params = getattr(cfg, "rope_parameters", None) if cfg is not None else None
                    # Avoid mutating model config; build an adapter config when rope_parameters is missing.
                    if cfg is not None and (not isinstance(rope_params, dict) or "rope_type" not in rope_params):
                        max_pos = getattr(cfg, "max_position_embeddings", 2048)
                        base = float(getattr(cfg, "rope_theta", 10000.0))
                        dim = getattr(cfg, "head_dim", None)
                        if dim is None:
                            hidden_size = getattr(cfg, "hidden_size", None)
                            num_heads = getattr(cfg, "num_attention_heads", 1) or 1
                            dim = hidden_size // num_heads if hidden_size else 128
                        compat_cfg = SimpleNamespace(
                            max_position_embeddings=max_pos,
                            rope_parameters={"rope_type": "default", "rope_theta": base},
                            head_dim=dim,
                            hidden_size=getattr(cfg, "hidden_size", dim),
                            num_attention_heads=getattr(cfg, "num_attention_heads", 1),
                        )
                        if args:
                            args = (compat_cfg,) + args[1:]
                        else:
                            kwargs["config"] = compat_cfg
                    return original_init(self, *args, **kwargs)

                patched_init._one_align_compat = True
                LlamaRotaryEmbedding.__init__ = patched_init

            original_forward = LlamaRotaryEmbedding.forward
            if not getattr(original_forward, "_one_align_compat", False):
                forward_sig = inspect.signature(original_forward)
                forward_params = list(forward_sig.parameters.keys())

                if "position_ids" in forward_params:

                    def patched_forward(self, x, position_ids=None, seq_len=None):
                        if seq_len is not None and position_ids is None:
                            batch_size = x.shape[0]
                            device = x.device
                            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
                            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                        return original_forward(self, x, position_ids=position_ids)

                else:

                    def patched_forward(self, x, position_ids=None, seq_len=None):
                        if seq_len is None and position_ids is not None:
                            seq_len = position_ids.shape[-1]
                        return original_forward(self, x, seq_len=seq_len)

                patched_forward._one_align_compat = True
                LlamaRotaryEmbedding.forward = patched_forward

            print("[OneAlign] 已修复 LlamaRotaryEmbedding 兼容性")

        except ImportError:
            pass
        except Exception as e:
            print(f"[OneAlign] 警告: 补丁失败 ({e})，可能导致兼容性问题")

    @staticmethod
    def _patch_cache_compatibility():
        """
        One-Align remote `modeling_llama2.py` uses:
        `from transformers.models.llama.modeling_llama import *`
        but transformers 5.x has a very small __all__, causing NameError.
        """
        try:
            from transformers.cache_utils import Cache

            try:
                import transformers.models.llama.modeling_llama as M

                if not hasattr(M, "Cache"):
                    M.Cache = Cache

                if hasattr(M, "__all__"):
                    extra = [
                        n
                        for n in (
                            "Cache",
                            "BaseModelOutputWithPast",
                            "CausalLMOutputWithPast",
                            "logger",
                            "LlamaRotaryEmbedding",
                            "apply_rotary_pos_emb",
                            "repeat_kv",
                            "LlamaMLP",
                            "LlamaRMSNorm",
                        )
                        if hasattr(M, n) and n not in M.__all__
                    ]
                    if extra:
                        M.__all__ = list(M.__all__) + extra
            except ImportError:
                pass

            try:
                import transformers.models.llama as llama_pkg

                if not hasattr(llama_pkg, "Cache"):
                    llama_pkg.Cache = Cache
            except ImportError:
                pass

            print("[OneAlign] 已修复 Cache 兼容性 (transformers 5.x)")
        except Exception as e:
            print(f"[OneAlign] 警告: Cache 补丁失败 ({e})")

    @staticmethod
    def _patch_model_config(config):
        """Patch old One-Align config fields for transformers 5.x."""
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            if rope_type in (None, "default"):
                # One-Align old code expects rope_scaling=None
                config.rope_scaling = None
            else:
                rope_scaling["type"] = rope_type
                if "factor" not in rope_scaling and "scaling_factor" in rope_scaling:
                    rope_scaling["factor"] = rope_scaling["scaling_factor"]
                config.rope_scaling = rope_scaling

        # Required by transformers 5.x LlamaMLP
        if not hasattr(config, "mlp_bias"):
            config.mlp_bias = False

        # Keep One-Align on eager attention path.
        config._attn_implementation = "eager"
        config.use_cache = False
        return config

    def _patch_loaded_model_compatibility(self):
        """Patch runtime attributes on the loaded model instance."""
        if not hasattr(self.model, "model"):
            return

        base_model = self.model.model

        if not hasattr(base_model, "_use_flash_attention_2"):
            base_model._use_flash_attention_2 = False
        if not hasattr(base_model, "_use_sdpa"):
            base_model._use_sdpa = False

        # transformers 5.x removed get_head_mask; One-Align visual abstractor still calls it.
        visual_abstractor = getattr(base_model, "visual_abstractor", None)
        if visual_abstractor is not None and not hasattr(visual_abstractor, "get_head_mask"):

            def _compat_get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked=False):
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            visual_abstractor.get_head_mask = _compat_get_head_mask.__get__(
                visual_abstractor, visual_abstractor.__class__
            )

    def load_model(self):
        """Load One-Align model."""
        if self.model is not None:
            return

        from transformers import AutoConfig, AutoModel

        self._patch_transformers_compatibility()
        self._patch_cache_compatibility()
        self._patch_llama_rotary_embedding()

        self.device = self._select_device()
        print(f"[OneAlign] 使用设备: {self.device}")
        print("[OneAlign] 正在加载模型 (首次约需 1-2 分钟)...")

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        config = self._patch_model_config(config)

        if self.device == "mps":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                config=config,
                dtype=torch.float16,
                device_map="mps",
                trust_remote_code=True,
            )
        elif self.device == "cuda":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                config=config,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                config=config,
                dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self._patch_loaded_model_compatibility()
        print("[OneAlign] 已修复 attention 兼容性")

        self.model.eval()
        print("[OneAlign] 模型加载成功")

    @staticmethod
    def _to_float_score(score_value) -> float:
        if isinstance(score_value, (list, tuple)):
            score_value = score_value[0]
        if torch.is_tensor(score_value):
            return float(score_value.detach().float().item())
        return float(score_value)

    def score_image(self, image_path: str) -> Dict:
        """
        Score a single image.

        Returns:
            {
                "quality": float,      # 0-100
                "aesthetic": float,    # 0-100
                "total": float,        # 0-100
                "rating": int,         # 0-4
                "pick_flag": str,      # "picked" / "rejected" / ""
                "color_label": str,    # "Green" / "Yellow" / "Red" / "Purple" / ""
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        if self.model is None:
            self.load_model()

        image = Image.open(image_path).convert("RGB")

        with torch.inference_mode():
            quality_score = self.model.score(
                [image],
                task_="quality",
                input_="image",
            )
            quality = self._to_float_score(quality_score) * 20

            aesthetic_score = self.model.score(
                [image],
                task_="aesthetics",
                input_="image",
            )
            aesthetic = self._to_float_score(aesthetic_score) * 20

        total = quality * self.quality_weight + aesthetic * self.aesthetic_weight
        rating, pick_flag, color_label = self._map_to_rating(total)

        return {
            "quality": quality,
            "aesthetic": aesthetic,
            "total": total,
            "rating": rating,
            "pick_flag": pick_flag,
            "color_label": color_label,
        }

    def score_batch(self, image_paths: List[str]) -> List[Dict]:
        """Score a list of images."""
        results = []
        for path in image_paths:
            try:
                result = self.score_image(path)
                result["file"] = path
                results.append(result)
            except Exception as e:
                results.append({"file": path, "error": str(e)})
        return results

    @staticmethod
    def _map_to_rating(total_score: float) -> Tuple[int, str, str]:
        """
        Map total score to rating/flags.
        5 stars are reserved for manual user curation.
        """
        t4, t3, t2, t1 = _thresholds

        if total_score >= t4:
            return 4, "", ""
        if total_score >= t3:
            return 3, "", ""
        if total_score >= t2:
            return 2, "", ""
        if total_score >= t1:
            return 1, "", ""
        return 0, "", ""

    def warmup(self):
        """Warmup model."""
        if self.model is None:
            self.load_model()


_thresholds = (78.0, 72.0, 66.0, 58.0)
_scorer_instance = None


def set_thresholds(t4: float, t3: float, t2: float, t1: float):
    """Set custom thresholds."""
    global _thresholds
    _thresholds = (t4, t3, t2, t1)


def get_one_align_scorer(
    model_path: Optional[str] = None,
    quality_weight: float = 0.4,
    aesthetic_weight: float = 0.6,
) -> OneAlignScorer:
    """Get singleton scorer."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = OneAlignScorer(
            model_path=model_path,
            quality_weight=quality_weight,
            aesthetic_weight=aesthetic_weight,
        )
    return _scorer_instance


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python one_align_scorer.py <图片路径>")
        sys.exit(1)

    scorer = get_one_align_scorer()
    result = scorer.score_image(sys.argv[1])

    print(f"\n{'=' * 50}")
    print("评分结果")
    print(f"{'=' * 50}")
    print(f"质量分: {result['quality']:.2f}")
    print(f"美学分: {result['aesthetic']:.2f}")
    print(f"综合分: {result['total']:.2f}")
    print(f"星级: {'*' * result['rating']} ({result['rating']}星)")
    print(f"旗标: {result['pick_flag'] or '-'}")
    print(f"色标: {result['color_label'] or '-'}")
