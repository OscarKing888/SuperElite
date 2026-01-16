# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct MLX 测试 (修正版)
测试用于替代 Co-Instruct 的图像描述和标题生成功能
"""

import time
import os
import json
from pathlib import Path

# 测试图片
TEST_IMAGE = "/Volumes/990PRO4TB/2025/2025-09-20/_Z8L1493.NEF"

print("📷 Qwen3-VL-8B-Instruct MLX 测试")
print("=" * 60)
print(f"测试图片: {TEST_IMAGE}")

# ==================== 计时开始 ====================
total_start = time.time()

# 步骤 1: 加载图片
print("\n⏱️ 步骤 1: 加载图片...")
step_start = time.time()

from PIL import Image
import rawpy
import io

with rawpy.imread(TEST_IMAGE) as raw:
    thumb = raw.extract_thumb()
    image = Image.open(io.BytesIO(thumb.data)).convert("RGB")

# 缩小图片
w, h = image.size
if max(w, h) > 672:
    if w > h:
        new_w, new_h = 672, int(h * 672 / w)
    else:
        new_h, new_w = 672, int(w * 672 / h)
    image = image.resize((new_w, new_h), Image.LANCZOS)

# 保存临时图片供 mlx-vlm 使用
temp_image_path = "/tmp/test_image_for_qwen.jpg"
image.save(temp_image_path, "JPEG", quality=95)

image_load_time = time.time() - step_start
print(f"   图片尺寸: {image.size}")
print(f"   ✅ 用时: {image_load_time:.2f}s")

# 步骤 2: 加载模型
print("\n⏱️ 步骤 2: 加载 Qwen3-VL-8B-Instruct MLX 模型...")
step_start = time.time()

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL_PATH = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"

# 加载模型和处理器
model, processor = load(MODEL_PATH)
config = load_config(MODEL_PATH)

model_load_time = time.time() - step_start
print(f"   ✅ 用时: {model_load_time:.2f}s")

# 步骤 3: 生成中文标题
print("\n⏱️ 步骤 3: 生成中文标题...")
step_start = time.time()

prompt_title = "为这张照片创作一个富有诗意的中文标题，5-10个字。只输出标题，不要其他内容。"
formatted_prompt = apply_chat_template(processor, config, prompt_title, num_images=1)

# 正确的 generate 调用方式：image 参数要单独传
title_response = generate(
    model, 
    processor, 
    formatted_prompt,
    image=[temp_image_path],  # 使用 image 关键字参数
    max_tokens=50,
    verbose=False
)

title_time = time.time() - step_start
print(f"   ✅ 用时: {title_time:.2f}s")

# 步骤 4: 生成详细描述
print("\n⏱️ 步骤 4: 生成详细画面描述...")
step_start = time.time()

prompt_desc = """请详细描述这张照片的画面内容，包括：
1. 主体是什么
2. 环境和背景
3. 光线条件
4. 色彩特点
5. 画面氛围和情感

用中文回答，尽可能详细。"""

formatted_prompt_desc = apply_chat_template(processor, config, prompt_desc, num_images=1)

desc_response = generate(
    model, 
    processor, 
    formatted_prompt_desc,
    image=[temp_image_path],
    max_tokens=300,
    verbose=False
)

desc_time = time.time() - step_start
print(f"   ✅ 用时: {desc_time:.2f}s")

# ==================== 结果汇总 ====================
total_time = time.time() - total_start

print("\n" + "=" * 60)
print("📌 生成结果")
print("=" * 60)
print(f"\n🏷️ 中文标题:")
print(f"   {title_response}")

print(f"\n📝 画面描述:")
print("-" * 40)
print(desc_response)

print("\n" + "=" * 60)
print("📊 用时统计")
print("=" * 60)
print(f"   图片加载:   {image_load_time:>6.2f}s")
print(f"   模型加载:   {model_load_time:>6.2f}s")
print(f"   标题生成:   {title_time:>6.2f}s")
print(f"   描述生成:   {desc_time:>6.2f}s")
print("-" * 30)
print(f"   总用时:     {total_time:>6.2f}s")
print("=" * 60)

# 清理临时文件
if os.path.exists(temp_image_path):
    os.remove(temp_image_path)
