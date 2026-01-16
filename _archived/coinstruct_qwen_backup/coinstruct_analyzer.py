# -*- coding: utf-8 -*-
"""
Co-Instruct 分析器
用于生成关键字、场景描述、标题等元数据
"""

import os
import time
import torch
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, Any

# 模型单例
_model = None
_model_loading = False


def get_model():
    """获取或加载 Co-Instruct 模型（单例）"""
    global _model, _model_loading
    
    if _model is not None:
        return _model
    
    if _model_loading:
        # 避免重复加载
        while _model_loading and _model is None:
            time.sleep(0.5)
        return _model
    
    _model_loading = True
    
    try:
        print("[Co-Instruct] 正在加载模型...")
        from transformers import AutoModelForCausalLM
        
        _model = AutoModelForCausalLM.from_pretrained(
            "q-future/co-instruct",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map={"": "mps"}
        )
        print("[Co-Instruct] 模型加载完成")
        return _model
    except Exception as e:
        print(f"[Co-Instruct] 模型加载失败: {e}")
        raise
    finally:
        _model_loading = False


def unload_model():
    """卸载模型释放内存"""
    global _model
    if _model is not None:
        del _model
        _model = None
        torch.mps.empty_cache()
        print("[Co-Instruct] 模型已卸载")


def prepare_image(image_path: str) -> Image.Image:
    """
    准备图片用于分析
    支持 RAW 和常规图片格式，RAW 使用内嵌预览图
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    
    # RAW 格式需要转换
    raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.orf', '.raf', '.rw2', '.dng', '.pef', '.raw'}
    
    if path.suffix.lower() in raw_extensions:
        # 检查是否有同名 JPG
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
            jpg_path = path.with_suffix(ext)
            if jpg_path.exists():
                return Image.open(jpg_path).convert("RGB")
        
        # 使用 rawpy 提取内嵌缩略图（更快）
        try:
            import rawpy
            import io
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return Image.open(io.BytesIO(thumb.data)).convert("RGB")
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    return Image.fromarray(thumb.data).convert("RGB")
                else:
                    # 无法提取缩略图，使用完整解码
                    rgb = raw.postprocess()
                    return Image.fromarray(rgb).convert("RGB")
        except ImportError:
            raise RuntimeError("需要安装 rawpy 处理 RAW 文件")
    else:
        return Image.open(image_path).convert("RGB")


def resize_for_analysis(image: Image.Image, max_size: int = 672) -> Image.Image:
    """调整图片大小以加快分析速度"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    return image.resize((new_w, new_h), Image.LANCZOS)


# ==================== Prompts ====================

PROMPTS = {
    "keywords_en": "USER: The image: <|image|> Generate 10 descriptive keywords for this photograph, separated by commas. ASSISTANT:",
    
    "keywords_cn": "USER: The image: <|image|> 为这张照片生成10个描述性关键词，用逗号分隔。 ASSISTANT:",
    
    "caption_en": "USER: The image: <|image|> Describe this photograph in 2-3 sentences. Focus on the subject, setting, lighting, and mood. ASSISTANT:",
    
    "caption_cn": "USER: The image: <|image|> 用2-3句话描述这张照片的场景和氛围。 ASSISTANT:",
    
    "title_en": "USER: The image: <|image|> Create a poetic title for this photograph in 3-6 words. ASSISTANT:",
    
    "title_cn": "USER: The image: <|image|> 为这张照片创作一个富有诗意的中文标题，5-10个字。 ASSISTANT:",
    
    "scene": "USER: The image: <|image|> Classify this photograph into one category: sunset, sunrise, mountain, ocean, forest, city, wildlife, portrait, street, architecture, night, aurora, waterfall, desert, lake, field, sky, abstract. Answer with one word only. ASSISTANT:",
    
    "mood": "USER: The image: <|image|> Describe the mood and atmosphere of this photograph in 2-3 words, such as: peaceful, dramatic, mysterious, romantic, melancholic, energetic, serene. ASSISTANT:",
}


def analyze(
    image_path: str,
    tasks: list = None,
    language: str = "cn"  # "cn" 或 "en"
) -> Dict[str, Any]:
    """
    分析图片，生成元数据
    
    Args:
        image_path: 图片路径
        tasks: 要执行的任务列表 ['keywords', 'caption', 'title', 'scene', 'mood']
               默认全部执行
        language: 语言偏好 "cn" 或 "en"
    
    Returns:
        {
            "success": True,
            "keywords": "...",
            "caption": "...",
            "title": "...",
            "scene": "...",
            "mood": "...",
            "processing_time": 12.5
        }
    """
    if tasks is None:
        tasks = ["keywords", "caption", "title", "scene", "mood"]
    
    start_time = time.time()
    result = {"success": False}
    
    try:
        # 加载模型
        model = get_model()
        
        # 准备图片
        image = prepare_image(image_path)
        image = resize_for_analysis(image)
        
        # 执行各项分析
        for task in tasks:
            prompt_key = task
            
            # 选择语言版本
            if task in ["keywords", "caption", "title"]:
                prompt_key = f"{task}_{language}"
            
            if prompt_key not in PROMPTS:
                continue
            
            prompt = PROMPTS[prompt_key]
            
            try:
                response = model.chat(prompt, [image], max_new_tokens=150)
                
                # 清理响应
                if isinstance(response, str):
                    response = response.strip()
                else:
                    response = str(response).strip()
                
                result[task] = response
                
            except Exception as e:
                result[task] = f"[Error: {e}]"
        
        result["success"] = True
        result["processing_time"] = round(time.time() - start_time, 2)
        
    except Exception as e:
        result["error"] = str(e)
        result["processing_time"] = round(time.time() - start_time, 2)
    
    return result


# ==================== 测试 ====================

if __name__ == "__main__":
    import sys
    
    # 查找测试图片
    test_dir = Path(__file__).parent.parent / "test_photos"
    test_image = None
    
    if test_dir.exists():
        for f in test_dir.iterdir():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                test_image = str(f)
                break
    
    if test_image is None:
        print("请提供测试图片路径作为参数")
        print("用法: python coinstruct_analyzer.py /path/to/image.jpg")
        sys.exit(1)
    
    print(f"📷 测试图片: {test_image}")
    print("=" * 60)
    
    result = analyze(test_image, language="cn")
    
    print("\n" + "=" * 60)
    print("📊 分析结果:")
    print("=" * 60)
    
    for key, value in result.items():
        print(f"\n🔹 {key}:")
        print(f"   {value}")
