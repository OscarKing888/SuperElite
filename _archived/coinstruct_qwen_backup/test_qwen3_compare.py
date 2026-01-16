# -*- coding: utf-8 -*-
"""
Qwen3-VL 4bit vs 8bit 对比测试
同一张图片测试两个版本
"""

import time
import os
from pathlib import Path
from PIL import Image

# 测试图片
TEST_IMAGE = "/Users/jameszhenyu/Desktop/NEWTEST_preprocessed_1024/4星/乌云盖顶马蹄湾-250214-8256 x 5504-F.jpg"

print("📷 Qwen3-VL 4bit vs 8bit 对比测试")
print("=" * 60)
print(f"测试图片: {Path(TEST_IMAGE).name}")

# 加载图片
print("\n⏱️ 加载图片...")
image = Image.open(TEST_IMAGE).convert("RGB")

# 缩小图片
w, h = image.size
if max(w, h) > 672:
    if w > h:
        new_w, new_h = 672, int(h * 672 / w)
    else:
        new_h, new_w = 672, int(w * 672 / h)
    image = image.resize((new_w, new_h), Image.LANCZOS)

# 保存临时图片
temp_image_path = "/tmp/test_image_compare.jpg"
image.save(temp_image_path, "JPEG", quality=95)
print(f"   图片尺寸: {image.size}")

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Prompts
prompt_title = "为这张照片创作一个富有诗意的中文标题，5-10个字。只输出标题，不要其他内容。"
prompt_desc = """请详细描述这张照片的画面内容，包括：
1. 主体是什么
2. 环境和背景
3. 光线条件
4. 色彩特点
5. 画面氛围和情感

用中文回答，尽可能详细。"""

results = {}

# ==================== 测试 8bit 版本 ====================
print("\n" + "=" * 60)
print("🔹 测试 8bit 版本")
print("=" * 60)

MODEL_8BIT = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-8bit"

step_start = time.time()
model_8bit, processor_8bit = load(MODEL_8BIT)
config_8bit = load_config(MODEL_8BIT)
load_8bit = time.time() - step_start
print(f"   模型加载: {load_8bit:.2f}s")

# 标题
formatted_prompt = apply_chat_template(processor_8bit, config_8bit, prompt_title, num_images=1)
step_start = time.time()
title_8bit = generate(model_8bit, processor_8bit, formatted_prompt, image=[temp_image_path], max_tokens=50, verbose=False)
title_time_8bit = time.time() - step_start
print(f"   标题生成: {title_time_8bit:.2f}s")

# 描述
formatted_prompt_desc = apply_chat_template(processor_8bit, config_8bit, prompt_desc, num_images=1)
step_start = time.time()
desc_8bit = generate(model_8bit, processor_8bit, formatted_prompt_desc, image=[temp_image_path], max_tokens=300, verbose=False)
desc_time_8bit = time.time() - step_start
print(f"   描述生成: {desc_time_8bit:.2f}s")

results['8bit'] = {
    'title': title_8bit.text if hasattr(title_8bit, 'text') else str(title_8bit),
    'desc': desc_8bit.text if hasattr(desc_8bit, 'text') else str(desc_8bit),
    'title_time': title_time_8bit,
    'desc_time': desc_time_8bit,
    'load_time': load_8bit
}

# 释放内存
del model_8bit, processor_8bit
import gc
gc.collect()

# ==================== 测试 4bit 版本 ====================
print("\n" + "=" * 60)
print("🔹 测试 4bit 版本")
print("=" * 60)

MODEL_4BIT = "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit"

step_start = time.time()
model_4bit, processor_4bit = load(MODEL_4BIT)
config_4bit = load_config(MODEL_4BIT)
load_4bit = time.time() - step_start
print(f"   模型加载: {load_4bit:.2f}s")

# 标题
formatted_prompt = apply_chat_template(processor_4bit, config_4bit, prompt_title, num_images=1)
step_start = time.time()
title_4bit = generate(model_4bit, processor_4bit, formatted_prompt, image=[temp_image_path], max_tokens=50, verbose=False)
title_time_4bit = time.time() - step_start
print(f"   标题生成: {title_time_4bit:.2f}s")

# 描述
formatted_prompt_desc = apply_chat_template(processor_4bit, config_4bit, prompt_desc, num_images=1)
step_start = time.time()
desc_4bit = generate(model_4bit, processor_4bit, formatted_prompt_desc, image=[temp_image_path], max_tokens=300, verbose=False)
desc_time_4bit = time.time() - step_start
print(f"   描述生成: {desc_time_4bit:.2f}s")

results['4bit'] = {
    'title': title_4bit.text if hasattr(title_4bit, 'text') else str(title_4bit),
    'desc': desc_4bit.text if hasattr(desc_4bit, 'text') else str(desc_4bit),
    'title_time': title_time_4bit,
    'desc_time': desc_time_4bit,
    'load_time': load_4bit
}

# ==================== 结果对比 ====================
print("\n" + "=" * 60)
print("📊 对比结果")
print("=" * 60)

print("\n🏷️ 中文标题:")
print(f"   8bit: {results['8bit']['title']}")
print(f"   4bit: {results['4bit']['title']}")

print("\n📝 画面描述 (8bit):")
print("-" * 40)
print(results['8bit']['desc'][:500] + "..." if len(results['8bit']['desc']) > 500 else results['8bit']['desc'])

print("\n📝 画面描述 (4bit):")
print("-" * 40)
print(results['4bit']['desc'][:500] + "..." if len(results['4bit']['desc']) > 500 else results['4bit']['desc'])

print("\n" + "=" * 60)
print("📊 速度对比")
print("=" * 60)
print(f"   {'指标':<12} {'8bit':<12} {'4bit':<12} {'4bit提升':<12}")
print("-" * 48)
print(f"   {'模型加载':<10} {results['8bit']['load_time']:<10.2f}s {results['4bit']['load_time']:<10.2f}s")
print(f"   {'标题生成':<10} {results['8bit']['title_time']:<10.2f}s {results['4bit']['title_time']:<10.2f}s {(1-results['4bit']['title_time']/results['8bit']['title_time'])*100:>+.0f}%")
print(f"   {'描述生成':<10} {results['8bit']['desc_time']:<10.2f}s {results['4bit']['desc_time']:<10.2f}s {(1-results['4bit']['desc_time']/results['8bit']['desc_time'])*100:>+.0f}%")
total_8bit = results['8bit']['title_time'] + results['8bit']['desc_time']
total_4bit = results['4bit']['title_time'] + results['4bit']['desc_time']
print(f"   {'推理总计':<10} {total_8bit:<10.2f}s {total_4bit:<10.2f}s {(1-total_4bit/total_8bit)*100:>+.0f}%")
print("=" * 60)

# 清理
if os.path.exists(temp_image_path):
    os.remove(temp_image_path)
