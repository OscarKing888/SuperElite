# -*- coding: utf-8 -*-
"""
Co-Instruct 功能测试脚本
逐一测试各项功能
"""

import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image

# 测试图片
TEST_IMAGE = "/Volumes/990PRO4TB/2025/2025-09-20/_Z8L1493.NEF"

# EXIF 信息（从 exiftool 读取）
EXIF_INFO = {
    "datetime": "2025-09-20 13:44:58",
    "exposure": "1/400s",
    "aperture": "f/9.0",
    "iso": "ISO 400",
    "focal_length": "280mm",
    "lens": "NIKKOR Z 100-400mm f/4.5-5.6 VR S",
    "camera": "Nikon Z8",
    "gps": None  # 无 GPS 信息
}

# ==================== 模型加载 ====================

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    
    print("[Co-Instruct] 正在加载模型...")
    from transformers import AutoModelForCausalLM
    
    _model = AutoModelForCausalLM.from_pretrained(
        "q-future/co-instruct",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": "mps"}
    )
    print("[Co-Instruct] 模型加载完成\n")
    return _model


def load_image(image_path: str) -> Image.Image:
    """加载图片，支持 RAW"""
    path = Path(image_path)
    
    # RAW 格式：提取内嵌缩略图
    raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.orf', '.raf', '.rw2', '.dng'}
    
    if path.suffix.lower() in raw_extensions:
        import rawpy
        import io
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                return Image.open(io.BytesIO(thumb.data)).convert("RGB")
            else:
                rgb = raw.postprocess()
                return Image.fromarray(rgb).convert("RGB")
    else:
        return Image.open(image_path).convert("RGB")


def resize_image(image: Image.Image, max_size: int = 672) -> Image.Image:
    """缩小图片"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w > h:
        new_w, new_h = max_size, int(h * max_size / w)
    else:
        new_h, new_w = max_size, int(w * max_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def ask_model(prompt: str, image: Image.Image, max_tokens: int = 300) -> str:
    """向模型提问"""
    model = get_model()
    response = model.chat(prompt, [image], max_new_tokens=max_tokens)
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


# ==================== 测试功能 ====================

def test_1_title(image):
    """功能1: 中文标题"""
    print("=" * 60)
    print("🔹 功能 1: 图片中文标题")
    print("=" * 60)
    
    prompt = "USER: The image: <|image|> 为这张照片创作一个富有诗意的中文标题，5-10个字，能体现画面的意境和氛围。 ASSISTANT:"
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=50)
    elapsed = time.time() - start
    
    print(f"\n📌 标题: {result}")
    print(f"⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_2_description(image):
    """功能2: 详细画面解读"""
    print("\n" + "=" * 60)
    print("🔹 功能 2: 图片中文解读（详细描述画面内容）")
    print("=" * 60)
    
    prompt = """USER: The image: <|image|> 
请详细描述这张照片的画面内容，包括：
1. 主体是什么
2. 环境和背景
3. 光线条件
4. 色彩特点
5. 画面氛围和情感

用中文回答，尽可能详细。 ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=500)
    elapsed = time.time() - start
    
    print(f"\n📝 解读:\n{result}")
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_3_keywords(image):
    """功能3: 关键字提取"""
    print("\n" + "=" * 60)
    print("🔹 功能 3: 关键字提取（不超过10个，要具体）")
    print("=" * 60)
    
    prompt = """USER: The image: <|image|> 
为这张照片生成不超过10个关键词，要求：
1. 具体而非抽象（例如用"白鹭"而非"鸟类"）
2. 涵盖主体、场景、氛围
3. 用中文，逗号分隔
ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=100)
    elapsed = time.time() - start
    
    print(f"\n🏷️ 关键字: {result}")
    print(f"⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_4_strengths(image):
    """功能4: 摄影优点"""
    print("\n" + "=" * 60)
    print("🔹 功能 4: 摄影师角度 - 照片优点")
    print("=" * 60)
    
    prompt = """USER: The image: <|image|> 
从专业摄影师的角度，分析这张照片的优点，包括：
- 构图
- 光线运用
- 色彩
- 时机把握
- 主体表现

用中文简洁列出主要优点。 ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=300)
    elapsed = time.time() - start
    
    print(f"\n✅ 优点:\n{result}")
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_5_weaknesses(image):
    """功能5: 摄影缺点"""
    print("\n" + "=" * 60)
    print("🔹 功能 5: 摄影师角度 - 照片缺点/可改进之处")
    print("=" * 60)
    
    prompt = """USER: The image: <|image|> 
从专业摄影师的角度，分析这张照片可以改进的地方，包括：
- 构图是否有问题
- 曝光是否准确
- 对焦是否锐利
- 背景是否干净
- 时机是否最佳

如果没有明显问题，请说明。用中文回答。 ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=300)
    elapsed = time.time() - start
    
    print(f"\n⚠️ 可改进:\n{result}")
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_6_postprocessing(image):
    """功能6: 后期处理建议"""
    print("\n" + "=" * 60)
    print("🔹 功能 6: 后期处理建议（Lightroom/Photoshop/Nik）")
    print("=" * 60)
    
    prompt = """USER: The image: <|image|> 
为这张照片提供具体的后期处理建议，使用 Lightroom、Photoshop 或 Nik Collection 插件。请给出：

1. Lightroom 基础调整建议（曝光、对比度、色温等）
2. 局部调整建议（如需要）
3. Photoshop 进阶处理（如需要）
4. Nik 插件使用建议（如 Color Efex、Silver Efex 等）

用中文具体说明每一步。 ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=600)
    elapsed = time.time() - start
    
    print(f"\n🎨 后期建议:\n{result}")
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    return result


def test_7_social_post(image, exif_info: dict):
    """功能7: 社交媒体推文"""
    print("\n" + "=" * 60)
    print("🔹 功能 7: 微信朋友圈/小红书风格推文")
    print("=" * 60)
    
    exif_str = f"""
拍摄时间: {exif_info['datetime']}
相机: {exif_info['camera']}
镜头: {exif_info['lens']}
参数: {exif_info['focal_length']}, {exif_info['aperture']}, {exif_info['exposure']}, {exif_info['iso']}
"""
    
    prompt = f"""USER: The image: <|image|> 
结合这张照片和以下拍摄信息，生成一条适合发布在微信朋友圈或小红书的推文：

{exif_str}

要求：
1. 文风自然、有感染力
2. 可以适当文艺但不要过于矫情
3. 可以包含拍摄心得或技术分享
4. 添加 3-5 个适合的 hashtag

用中文。 ASSISTANT:"""
    
    start = time.time()
    result = ask_model(prompt, image, max_tokens=400)
    elapsed = time.time() - start
    
    print(f"\n📱 推文:\n{result}")
    print(f"\n⏱️ 耗时: {elapsed:.1f}s")
    return result


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("📷 Co-Instruct 功能测试")
    print("=" * 60)
    print(f"\n测试图片: {TEST_IMAGE}")
    
    # 检查文件
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 文件不存在: {TEST_IMAGE}")
        return
    
    # 加载图片
    print("\n正在加载图片...")
    image = load_image(TEST_IMAGE)
    image = resize_image(image)
    print(f"图片尺寸: {image.size}")
    
    # 加载模型
    get_model()
    
    # 选择测试
    print("\n" + "=" * 60)
    print("请选择要测试的功能:")
    print("  1 - 中文标题")
    print("  2 - 详细画面解读")
    print("  3 - 关键字提取")
    print("  4 - 摄影优点分析")
    print("  5 - 摄影缺点分析")
    print("  6 - 后期处理建议")
    print("  7 - 社交媒体推文")
    print("  a - 全部测试")
    print("  q - 退出")
    print("=" * 60)
    
    while True:
        choice = input("\n输入选项 (1-7, a, q): ").strip().lower()
        
        if choice == 'q':
            print("退出")
            break
        elif choice == '1':
            test_1_title(image)
        elif choice == '2':
            test_2_description(image)
        elif choice == '3':
            test_3_keywords(image)
        elif choice == '4':
            test_4_strengths(image)
        elif choice == '5':
            test_5_weaknesses(image)
        elif choice == '6':
            test_6_postprocessing(image)
        elif choice == '7':
            test_7_social_post(image, EXIF_INFO)
        elif choice == 'a':
            test_1_title(image)
            test_2_description(image)
            test_3_keywords(image)
            test_4_strengths(image)
            test_5_weaknesses(image)
            test_6_postprocessing(image)
            test_7_social_post(image, EXIF_INFO)
            print("\n" + "=" * 60)
            print("✅ 全部测试完成")
            print("=" * 60)
        else:
            print("无效选项")


if __name__ == "__main__":
    main()
