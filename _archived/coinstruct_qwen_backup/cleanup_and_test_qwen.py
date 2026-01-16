# -*- coding: utf-8 -*-
"""
清理 Co-Instruct 模型并测试 Qwen VL
"""

import os
import shutil
from pathlib import Path

# 1. 查找并显示 Co-Instruct 模型大小
print("=" * 60)
print("📦 检查 HuggingFace 缓存中的 Co-Instruct 模型")
print("=" * 60)

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
coinstruct_dirs = list(cache_dir.glob("*co-instruct*"))
mplug_dirs = list(cache_dir.glob("*mplug*"))

all_dirs = coinstruct_dirs + mplug_dirs

if all_dirs:
    total_size = 0
    for d in all_dirs:
        if d.is_dir():
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            total_size += size
            print(f"   {d.name}: {size_gb:.2f} GB")
    
    print(f"\n   总计: {total_size / (1024**3):.2f} GB")
    print("\n⚠️ 要删除这些模型，请运行: python cleanup_and_test_qwen.py --delete")
else:
    print("   未找到 Co-Instruct 相关模型缓存")

# 2. 处理删除参数
import sys
if "--delete" in sys.argv:
    print("\n" + "=" * 60)
    print("🗑️ 删除 Co-Instruct 模型...")
    print("=" * 60)
    for d in all_dirs:
        if d.is_dir():
            print(f"   删除: {d.name}")
            shutil.rmtree(d)
    print("   ✅ 删除完成！")

# 3. 显示 Qwen VL 模型信息
print("\n" + "=" * 60)
print("🔍 Qwen3-VL 模型信息")
print("=" * 60)
print("""
推荐模型（MLX 版本，针对 Apple Silicon 优化）:
  
  1. Qwen/Qwen2.5-VL-7B-Instruct (标准版)
     - 大小: ~15GB
     - 需要 MLX 格式转换
  
  2. mlx-community/Qwen2.5-VL-7B-Instruct-8bit (量化版)
     - 大小: ~8GB  
     - 已经是 MLX 格式，可直接使用
  
  3. mlx-community/Qwen2.5-VL-7B-Instruct-4bit (高度量化)
     - 大小: ~4GB
     - 速度最快，但质量略有下降

建议: 先尝试 8bit 版本，在速度和质量之间取得平衡。
""")
