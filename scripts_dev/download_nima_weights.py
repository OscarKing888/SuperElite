# -*- coding: utf-8 -*-
"""
下载 NIMA 美学权重到项目 models 目录。
来源: HuggingFace chaofengc/IQA-PyTorch-Weights (PyIQA)
对应: inception_resnet_v2-ava -> NIMA_InceptionV2_ava-b0c77c00.pth，保存为 nima_ava.pth
"""

import os
import shutil
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TARGET_FILE = MODELS_DIR / "nima_ava.pth"
REPO_ID = "chaofengc/IQA-PyTorch-Weights"
FILE_NAME = "NIMA_InceptionV2_ava-b0c77c00.pth"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists():
        print(f"已存在: {TARGET_FILE}")
        return 0

    # 可选：使用国内镜像
    env = os.environ.get("HF_ENDPOINT")
    if env:
        print(f"使用 HF 端点: {env}")

    print(f"下载: {REPO_ID} -> {FILE_NAME}")
    print(f"保存为: {TARGET_FILE}")
    print("(约 218 MB，请稍候...)")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        return 1

    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILE_NAME,
            local_dir=str(MODELS_DIR),
            force_download=False,
        )
        downloaded = Path(path)
        # 复制到固定文件名，供 pyiqa_scorer 使用
        if downloaded.resolve() != TARGET_FILE.resolve():
            shutil.copy2(downloaded, TARGET_FILE)
            print(f"已复制到: {TARGET_FILE}")
        if not TARGET_FILE.exists():
            raise FileNotFoundError(f"未找到: {TARGET_FILE}")
        print("完成.")
        return 0
    except Exception as e:
        print(f"下载失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
