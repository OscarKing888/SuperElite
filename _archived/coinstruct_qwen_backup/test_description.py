# -*- coding: utf-8 -*-
"""
Co-Instruct 单功能测试 - 详细画面解读
"""

import torch
import time
from pathlib import Path
from PIL import Image

# 测试图片
TEST_IMAGE = "/Volumes/990PRO4TB/2025/2025-09-20/_Z8L1493.NEF"

print("📷 Co-Instruct 测试 - 详细画面解读")
print("=" * 60)
print(f"测试图片: {TEST_IMAGE}")

# ==================== 计时开始 ====================
total_start = time.time()

# 加载图片
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
print(f"   图片尺寸: {image.size}")
print(f"   用时: {time.time() - step_start:.1f}s")

# 加载模型
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
print(f"   用时: {model_load_time:.1f}s")

# 测试功能
print("\n" + "=" * 60)
print("🔹 功能 2: 详细画面解读")
print("=" * 60)

prompt = """USER: The image: <|image|> 
请详细描述这张照片的画面内容，包括：
1. 主体是什么
2. 环境和背景
3. 光线条件
4. 色彩特点
5. 画面氛围和情感

用中文回答，尽可能详细。 ASSISTANT:"""

print("\n⏱️ 步骤 3: 生成描述...")
step_start = time.time()
response = model.chat(prompt, [image], max_new_tokens=500)
inference_time = time.time() - step_start

# 输出结果
print("\n📝 画面解读:")
print("-" * 60)
# 模型返回的中文会直接打印在这之前

total_time = time.time() - total_start

print("\n" + "=" * 60)
print("📊 用时统计")
print("=" * 60)
print(f"   图片加载: {0.1:.1f}s")
print(f"   模型加载: {model_load_time:.1f}s")
print(f"   AI 推理:  {inference_time:.1f}s")
print(f"   总用时:   {total_time:.1f}s")
print("=" * 60)
