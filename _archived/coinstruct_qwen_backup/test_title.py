# -*- coding: utf-8 -*-
"""
Co-Instruct 单功能测试 - 中文标题
"""

import torch
from pathlib import Path
from PIL import Image

# 测试图片
TEST_IMAGE = "/Volumes/990PRO4TB/2025/2025-09-20/_Z8L1493.NEF"

print("📷 Co-Instruct 测试 - 中文标题生成")
print("=" * 50)
print(f"测试图片: {TEST_IMAGE}")

# 加载图片
print("\n正在加载图片...")
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
print(f"图片尺寸: {image.size}")

# 加载模型
print("\n正在加载模型...")
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "q-future/co-instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map={"": "mps"}
)
print("模型加载完成")

# 测试中文标题
print("\n" + "=" * 50)
print("🔹 测试: 中文标题生成")
print("=" * 50)

import time
prompt = "USER: The image: <|image|> 为这张照片创作一个富有诗意的中文标题，5-10个字。 ASSISTANT:"

start = time.time()
response = model.chat(prompt, [image], max_new_tokens=50)
elapsed = time.time() - start

print(f"\n📌 响应类型: {type(response)}")
print(f"📌 响应内容: {response}")
print(f"⏱️ 耗时: {elapsed:.1f}s")
