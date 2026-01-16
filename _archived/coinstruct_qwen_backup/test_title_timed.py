# -*- coding: utf-8 -*-
"""
Co-Instruct 单功能测试 - 中文标题（详细计时版）
"""

import torch
import time
from pathlib import Path
from PIL import Image

# 测试图片
TEST_IMAGE = "/Volumes/990PRO4TB/2025/2025-09-20/_Z8L1493.NEF"

print("📷 Co-Instruct 测试 - 中文标题生成（详细计时）")
print("=" * 60)
print(f"测试图片: {TEST_IMAGE}")

# ==================== 计时开始 ====================
total_start = time.time()

# 步骤 1: 加载图片
print("\n⏱️ 步骤 1: 加载图片...")
step_start = time.time()
import rawpy
import io
with rawpy.imread(TEST_IMAGE) as raw:
    thumb = raw.extract_thumb()
    image = Image.open(io.BytesIO(thumb.data)).convert("RGB")

# 缩小
w, h = image.size
if max(w, h) > 672:
    if w > h:
        new_w, new_h = 672, int(h * 672 / w)
    else:
        new_h, new_w = 672, int(w * 672 / h)
    image = image.resize((new_w, new_h), Image.LANCZOS)
image_load_time = time.time() - step_start
print(f"   图片尺寸: {image.size}")
print(f"   ✅ 用时: {image_load_time:.2f}s")

# 步骤 2: 加载模型
print("\n⏱️ 步骤 2: 加载模型...")
step_start = time.time()
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "q-future/co-instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map={"": "mps"}
)
model_load_time = time.time() - step_start
print(f"   ✅ 用时: {model_load_time:.2f}s")

# 步骤 3: 生成标题
print("\n⏱️ 步骤 3: 生成中文标题...")
step_start = time.time()
prompt = "USER: The image: <|image|> 为这张照片创作一个富有诗意的中文标题，5-10个字。 ASSISTANT:"
response = model.chat(prompt, [image], max_new_tokens=30)
inference_time = time.time() - step_start
print(f"   ✅ 用时: {inference_time:.2f}s")

# ==================== 结果汇总 ====================
total_time = time.time() - total_start

print("\n" + "=" * 60)
print("📌 生成结果")
print("=" * 60)
print(f"   标题: {response}")

print("\n" + "=" * 60)
print("📊 用时统计")
print("=" * 60)
print(f"   图片加载:   {image_load_time:>6.2f}s")
print(f"   模型加载:   {model_load_time:>6.2f}s")
print(f"   AI 推理:    {inference_time:>6.2f}s")
print("-" * 30)
print(f"   总用时:     {total_time:>6.2f}s")
print("=" * 60)
