# -*- coding: utf-8 -*-
"""
摄影评片模块
使用 VLM 模型进行专业摄影评价
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
from dataclasses import dataclass
from enum import Enum


class DetailLevel(Enum):
    """评片详细程度"""
    BRIEF = "brief"       # 简约 100-200 字
    NORMAL = "normal"     # 正常 300-400 字
    DETAILED = "detailed" # 详细 ≤500 字


@dataclass
class CritiqueConfig:
    """评片配置"""
    detail_level: DetailLevel = DetailLevel.NORMAL
    enable_title: bool = False
    enable_keywords: bool = False
    enable_scene: bool = False
    enable_critique: bool = True
    enable_exif_analysis: bool = True  # 默认开启


# 模型单例
_model = None
_processor = None
_config = None


def get_model():
    """获取或加载模型（单例）"""
    global _model, _processor, _config
    
    if _model is not None:
        return _model, _processor, _config
    
    print("[摄影评片] 正在加载模型...")
    start = time.time()
    
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    
    MODEL_PATH = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"
    
    _model, _processor = load(MODEL_PATH)
    _config = load_config(MODEL_PATH)
    
    print(f"[摄影评片] 模型加载完成 ({time.time() - start:.1f}s)")
    return _model, _processor, _config


def unload_model():
    """卸载模型释放内存"""
    global _model, _processor, _config
    if _model is not None:
        del _model, _processor, _config
        _model = _processor = _config = None
        import gc
        gc.collect()
        print("[摄影评片] 模型已卸载")


def prepare_image(image_path: str, max_size: int = 1024) -> str:
    """
    准备图片用于分析，返回临时文件路径
    支持 RAW 和常规图片格式
    统一处理为长边 1024px
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    # RAW 格式
    raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.orf', '.raf', '.rw2', '.dng', '.pef', '.raw'}
    
    if path.suffix.lower() in raw_extensions:
        # 检查同名 JPG
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            jpg_path = path.with_suffix(ext)
            if jpg_path.exists():
                image = Image.open(jpg_path).convert("RGB")
                break
        else:
            # 使用 rawpy 提取缩略图
            import rawpy
            import io
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    image = Image.open(io.BytesIO(thumb.data)).convert("RGB")
                else:
                    rgb = raw.postprocess()
                    image = Image.fromarray(rgb).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    
    # 调整大小：长边 1024px
    w, h = image.size
    if max(w, h) > max_size:
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_h, new_w = max_size, int(w * max_size / h)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 保存临时文件
    temp_path = f"/tmp/photo_critic_{os.getpid()}.jpg"
    image.save(temp_path, "JPEG", quality=95)
    
    return temp_path


def extract_exif(image_path: str) -> Dict[str, Any]:
    """
    提取 EXIF 信息
    返回曝光参数、器材、时间、GPS 等
    """
    exif_data = {}
    
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        
        image = Image.open(image_path)
        exif = image._getexif()
        
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                
                if tag == "Make":
                    exif_data["camera_make"] = str(value)
                elif tag == "Model":
                    exif_data["camera_model"] = str(value)
                elif tag == "LensModel":
                    exif_data["lens"] = str(value)
                elif tag == "FocalLength":
                    exif_data["focal_length"] = f"{value}mm" if isinstance(value, (int, float)) else str(value)
                elif tag == "FNumber":
                    exif_data["aperture"] = f"f/{value}" if isinstance(value, (int, float)) else str(value)
                elif tag == "ExposureTime":
                    if isinstance(value, tuple):
                        exif_data["shutter_speed"] = f"{value[0]}/{value[1]}s"
                    else:
                        exif_data["shutter_speed"] = f"{value}s"
                elif tag == "ISOSpeedRatings":
                    exif_data["iso"] = f"ISO {value}"
                elif tag == "DateTimeOriginal":
                    exif_data["datetime"] = str(value)
                elif tag == "GPSInfo":
                    # 简单 GPS 处理
                    try:
                        gps = {}
                        for key in value.keys():
                            decode = GPSTAGS.get(key, key)
                            gps[decode] = value[key]
                        if "GPSLatitude" in gps and "GPSLongitude" in gps:
                            exif_data["gps"] = True
                    except:
                        pass
    except Exception as e:
        exif_data["_error"] = str(e)
    
    return exif_data


def format_exif_context(exif_data: Dict[str, Any]) -> str:
    """格式化 EXIF 信息为 prompt 上下文"""
    if not exif_data or "_error" in exif_data:
        return ""
    
    parts = []
    
    if "camera_make" in exif_data and "camera_model" in exif_data:
        parts.append(f"相机: {exif_data['camera_make']} {exif_data['camera_model']}")
    if "lens" in exif_data:
        parts.append(f"镜头: {exif_data['lens']}")
    if "focal_length" in exif_data:
        parts.append(f"焦距: {exif_data['focal_length']}")
    if "aperture" in exif_data:
        parts.append(f"光圈: {exif_data['aperture']}")
    if "shutter_speed" in exif_data:
        parts.append(f"快门: {exif_data['shutter_speed']}")
    if "iso" in exif_data:
        parts.append(f"{exif_data['iso']}")
    if "datetime" in exif_data:
        parts.append(f"拍摄时间: {exif_data['datetime']}")
    if exif_data.get("gps"):
        parts.append("(含GPS信息)")
    
    if parts:
        return "拍摄参数: " + ", ".join(parts)
    return ""


def read_one_align_scores(image_path: str) -> Dict[str, Any]:
    """
    从 IPTC 读取 One-Align 评分
    City = 质量分, Province-State = 美学分, Rating = 星级
    
    Returns:
        {
            "quality": float or None,
            "aesthetic": float or None,
            "rating": int or None,
            "has_scores": bool
        }
    """
    scores = {
        "quality": None,
        "aesthetic": None,
        "rating": None,
        "has_scores": False
    }
    
    try:
        from exif_writer import get_exif_writer
        writer = get_exif_writer()
        
        # 读取 IPTC 字段
        result = writer.read_metadata(image_path)
        
        if result:
            # City = 质量分
            if "City" in result:
                try:
                    scores["quality"] = float(result["City"])
                except:
                    pass
            
            # Province-State = 美学分
            if "Province-State" in result:
                try:
                    scores["aesthetic"] = float(result["Province-State"])
                except:
                    pass
            
            # Rating = 星级
            if "Rating" in result:
                try:
                    scores["rating"] = int(result["Rating"])
                except:
                    pass
        
        # 检查是否有完整评分
        scores["has_scores"] = all([
            scores["quality"] is not None,
            scores["aesthetic"] is not None,
            scores["rating"] is not None
        ])
        
    except Exception as e:
        scores["_error"] = str(e)
    
    return scores


def get_one_align_scores(image_path: str) -> Dict[str, Any]:
    """
    获取 One-Align 评分
    如果 IPTC 中已有评分则直接返回，否则调用 One-Align 模型获取
    
    Returns:
        {
            "quality": float,
            "aesthetic": float,
            "total": float,
            "rating": int,
            "source": "iptc" | "one_align"
        }
    """
    # 先尝试从 IPTC 读取
    existing = read_one_align_scores(image_path)
    
    if existing["has_scores"]:
        # 计算综合分 (40% 质量 + 60% 美学)
        total = existing["quality"] * 0.4 + existing["aesthetic"] * 0.6
        return {
            "quality": existing["quality"],
            "aesthetic": existing["aesthetic"],
            "total": total,
            "rating": existing["rating"],
            "source": "iptc"
        }
    
    # 需要调用 One-Align 获取评分
    print("[摄影评片] 未找到评分，调用 One-Align 模型...")
    
    try:
        from one_align_scorer import get_one_align_scorer
        
        scorer = get_one_align_scorer()
        result = scorer.score_image(image_path)
        
        return {
            "quality": result["quality"],
            "aesthetic": result["aesthetic"],
            "total": result["total"],
            "rating": result["rating"],
            "source": "one_align"
        }
    except Exception as e:
        print(f"[摄影评片] One-Align 评分失败: {e}")
        return {
            "quality": None,
            "aesthetic": None,
            "total": None,
            "rating": None,
            "source": "error",
            "_error": str(e)
        }


def format_scores_context(scores: Dict[str, Any]) -> str:
    """
    格式化评分信息为 prompt 上下文
    让评片模型了解图片的质量和美学水平
    """
    if not scores or scores.get("source") == "error":
        return ""
    
    parts = []
    
    if scores.get("quality") is not None:
        quality = scores["quality"]
        if quality >= 80:
            level = "优秀"
        elif quality >= 70:
            level = "良好"
        elif quality >= 60:
            level = "中等"
        else:
            level = "一般"
        parts.append(f"技术质量分 {quality:.1f}/100 ({level})")
    
    if scores.get("aesthetic") is not None:
        aesthetic = scores["aesthetic"]
        if aesthetic >= 80:
            level = "出色"
        elif aesthetic >= 70:
            level = "不错"
        elif aesthetic >= 60:
            level = "中等"
        else:
            level = "一般"
        parts.append(f"美学评分 {aesthetic:.1f}/100 ({level})")
    
    if scores.get("rating") is not None:
        rating = scores["rating"]
        stars = "⭐" * rating if rating > 0 else "无星"
        parts.append(f"综合评级 {rating}星 ({stars})")
    
    if parts:
        return "AI 初评: " + ", ".join(parts)
    return ""


# ==================== Prompts ====================

DETAIL_SETTINGS = {
    DetailLevel.BRIEF: ("简要点评100-200字", 250),
    DetailLevel.NORMAL: ("正常点评300-400字", 500),
    DetailLevel.DETAILED: ("详细点评不超过500字", 650),
}

CRITIQUE_TEMPLATE = """请以专业摄影师视角评价这张照片：
{exif_context}
{scores_context}

1. 技术点评：曝光/对焦/构图
2. 艺术点评：情绪/光影/色彩
3. 主要优点(1-2点)
4. 需改进处(1-2点)
5. 后期建议

{detail_instruction}，用中文专业简洁回答。"""

TITLE_PROMPT = "为这张照片创作一个富有诗意的中文标题，5-10个字。只输出标题。"

KEYWORDS_PROMPT = """列出这张照片中能看到的关键元素，不超过10个词，用逗号分隔。
要求：具体可见事物，避免抽象名词（如"美丽"、"自然"等）。"""

SCENE_PROMPT = """判断这张照片的场景类型，从以下选项中选择一个：
风光、人像、街拍、建筑、野生动物、静物、美食、运动、新闻纪实、婚礼、商业产品、其他
只输出一个分类名称。"""


def generate(model, processor, config, prompt: str, image_path: str, max_tokens: int) -> str:
    """调用模型生成"""
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    
    response = mlx_generate(
        model,
        processor,
        formatted_prompt,
        image=[image_path],
        max_tokens=max_tokens,
        verbose=False
    )
    
    return response.text if hasattr(response, 'text') else str(response)


def critique(
    image_path: str,
    config: CritiqueConfig = None
) -> Dict[str, Any]:
    """
    摄影评片主函数
    
    Args:
        image_path: 图片路径
        config: 评片配置
    
    Returns:
        {
            "success": True,
            "critique": "...",      # 摄影评片
            "title": "...",         # 中文标题 (可选)
            "keywords": "...",      # 关键词 (可选)
            "scene": "...",         # 场景分类 (可选)
            "exif": {...},          # EXIF 信息
            "processing_time": 12.5
        }
    """
    if config is None:
        config = CritiqueConfig()
    
    start_time = time.time()
    result = {"success": False}
    temp_path = None
    
    try:
        # 获取模型
        model, processor, model_config = get_model()
        
        # 准备图片
        temp_path = prepare_image(image_path)
        
        # 提取 EXIF
        if config.enable_exif_analysis:
            exif_data = extract_exif(image_path)
            result["exif"] = exif_data
            exif_context = format_exif_context(exif_data)
        else:
            exif_context = ""
        
        # 获取 One-Align 评分（必要时触发评分）
        scores = get_one_align_scores(image_path)
        result["scores"] = scores
        scores_context = format_scores_context(scores)
        
        # 摄影评片
        if config.enable_critique:
            detail_instruction, max_tokens = DETAIL_SETTINGS[config.detail_level]
            prompt = CRITIQUE_TEMPLATE.format(
                exif_context=exif_context,
                scores_context=scores_context,
                detail_instruction=detail_instruction
            )
            result["critique"] = generate(model, processor, model_config, prompt, temp_path, max_tokens)
        
        # 中文标题
        if config.enable_title:
            result["title"] = generate(model, processor, model_config, TITLE_PROMPT, temp_path, 50)
        
        # 关键词
        if config.enable_keywords:
            result["keywords"] = generate(model, processor, model_config, KEYWORDS_PROMPT, temp_path, 100)
        
        # 场景分类
        if config.enable_scene:
            result["scene"] = generate(model, processor, model_config, SCENE_PROMPT, temp_path, 20)
        
        result["success"] = True
        result["processing_time"] = round(time.time() - start_time, 2)
        
    except Exception as e:
        result["error"] = str(e)
        result["processing_time"] = round(time.time() - start_time, 2)
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    
    return result


# ==================== 测试 ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python photo_critic.py <图片路径>")
        sys.exit(1)
    
    test_image = sys.argv[1]
    print(f"📷 测试图片: {test_image}")
    print("=" * 60)
    
    # 完整测试
    config = CritiqueConfig(
        detail_level=DetailLevel.NORMAL,
        enable_title=True,
        enable_keywords=True,
        enable_scene=True,
        enable_critique=True,
        enable_exif_analysis=True
    )
    
    result = critique(test_image, config)
    
    print("\n" + "=" * 60)
    print("📊 评片结果:")
    print("=" * 60)
    
    # 显示 One-Align 评分
    if result.get("scores"):
        scores = result["scores"]
        print(f"\n⭐ One-Align 评分 (来源: {scores.get('source', 'unknown')}):")
        if scores.get("quality") is not None:
            print(f"   技术质量: {scores['quality']:.1f}/100")
        if scores.get("aesthetic") is not None:
            print(f"   美学评分: {scores['aesthetic']:.1f}/100")
        if scores.get("total") is not None:
            print(f"   综合分: {scores['total']:.1f}/100")
        if scores.get("rating") is not None:
            print(f"   星级: {'⭐' * scores['rating']} ({scores['rating']}星)")
    
    if result.get("exif"):
        print(f"\n📸 EXIF 信息:")
        for k, v in result["exif"].items():
            if not k.startswith("_"):
                print(f"   {k}: {v}")
    
    if result.get("title"):
        print(f"\n🏷️ 标题: {result['title']}")
    
    if result.get("keywords"):
        print(f"\n🔑 关键词: {result['keywords']}")
    
    if result.get("scene"):
        print(f"\n🎬 场景: {result['scene']}")
    
    if result.get("critique"):
        print(f"\n📝 摄影评片:")
        print("-" * 40)
        print(result["critique"])
    
    print(f"\n⏱️ 处理时间: {result.get('processing_time', 0)}s")
