# -*- coding: utf-8 -*-
"""
修复 Qwen3-VL preprocessor 配置
"""

import json
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

# Qwen3-VL 模型
qwen3_dirs = list(cache_dir.glob("models--lmstudio-community--Qwen3-VL-8B-Instruct-MLX-8bit"))

for model_dir in qwen3_dirs:
    snapshots = list((model_dir / "snapshots").iterdir())
    for snapshot_dir in snapshots:
        preprocessor_file = snapshot_dir / "preprocessor_config.json"
        
        if preprocessor_file.exists():
            print(f"📝 修改: {preprocessor_file}")
            with open(preprocessor_file, 'r') as f:
                config = json.load(f)
            
            print(f"   原配置: {config.get('image_processor_type')}")
            
            # 修改为标准类型
            config["image_processor_type"] = "Qwen2VLImageProcessor"
            if "processor_class" in config:
                config["processor_class"] = "Qwen2VLProcessor"
            
            with open(preprocessor_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"   新配置: {config.get('image_processor_type')}")
            print("   ✅ 修复完成!")

# Qwen2.5-VL 模型
qwen25_dirs = list(cache_dir.glob("models--mlx-community--Qwen2.5-VL-*"))

for model_dir in qwen25_dirs:
    snapshots = list((model_dir / "snapshots").iterdir())
    for snapshot_dir in snapshots:
        preprocessor_file = snapshot_dir / "preprocessor_config.json"
        
        if preprocessor_file.exists():
            print(f"\n📝 修改: {preprocessor_file}")
            with open(preprocessor_file, 'r') as f:
                config = json.load(f)
            
            print(f"   原配置: {config.get('image_processor_type')}")
            
            # 修改为标准类型
            if "Fast" in str(config.get("image_processor_type", "")):
                config["image_processor_type"] = "Qwen2VLImageProcessor"
                with open(preprocessor_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"   新配置: {config.get('image_processor_type')}")
                print("   ✅ 修复完成!")

print("\n✅ 所有配置修复完毕!")
