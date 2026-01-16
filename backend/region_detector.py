# -*- coding: utf-8 -*-
"""
SuperElite - 区域检测与模型管理
根据用户 IP 自动选择 HuggingFace 下载源
"""

import os
import psutil
from pathlib import Path
from typing import Tuple, Optional


# ==================== 系统检测 ====================

def get_system_memory_gb() -> float:
    """获取系统总内存 (GB)"""
    return psutil.virtual_memory().total / (1024 ** 3)


def check_system_requirements() -> Tuple[bool, str]:
    """
    检查系统要求
    
    Returns:
        (通过, 错误信息)
    """
    memory_gb = get_system_memory_gb()
    min_memory = 16.0
    
    if memory_gb < min_memory:
        return False, f"系统内存不足: {memory_gb:.1f}GB (需要 ≥{min_memory:.0f}GB)"
    
    return True, ""


# ==================== 区域检测 ====================

def is_china_mainland(timeout: float = 3.0) -> bool:
    """
    检测用户是否在中国大陆
    
    Args:
        timeout: 请求超时时间 (秒)
    
    Returns:
        是否在中国大陆
    """
    try:
        import requests
        # 使用 ip-api.com 免费 API
        response = requests.get(
            "http://ip-api.com/json/?fields=countryCode",
            timeout=timeout
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("countryCode") == "CN"
    except Exception:
        pass
    
    return False  # 网络错误时默认使用官方源


def get_recommended_endpoint() -> Tuple[str, str, bool]:
    """
    获取推荐的下载源
    
    Returns:
        (endpoint_url, display_name, is_china)
    """
    is_china = is_china_mainland()
    
    if is_china:
        return (
            "https://hf-mirror.com",
            "国内镜像 (hf-mirror.com)",
            True
        )
    else:
        return (
            "https://huggingface.co",
            "官方源 (huggingface.co)",
            False
        )


def setup_hf_endpoint(endpoint: str = None):
    """
    设置 HuggingFace 下载源
    
    Args:
        endpoint: 下载源 URL，如果为 None 则自动检测
    """
    if endpoint is None:
        endpoint, name, _ = get_recommended_endpoint()
        print(f"[Region] 检测下载源: {name}")
    
    if "hf-mirror" in endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"[Region] 使用镜像: {endpoint}")
    else:
        # 官方源不需要设置
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]
        print(f"[Region] 使用官方源")


# ==================== 模型缓存检测 ====================

MODEL_ID = "q-future/one-align"


def get_model_cache_path() -> Path:
    """获取模型缓存路径"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HuggingFace 缓存目录格式: models--{org}--{name}
    model_dir_name = f"models--{MODEL_ID.replace('/', '--')}"
    return cache_dir / model_dir_name


def is_model_cached() -> bool:
    """检查模型是否已下载"""
    cache_path = get_model_cache_path()
    
    if not cache_path.exists():
        return False
    
    # 检查是否有 snapshots 目录且里面有文件
    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False
    
    # 至少有一个快照目录
    for snapshot in snapshots_dir.iterdir():
        if snapshot.is_dir():
            # 检查是否有模型文件
            files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
            if files:
                return True
    
    return False


def get_model_cache_size_gb() -> float:
    """获取模型缓存大小 (GB)"""
    cache_path = get_model_cache_path()
    
    if not cache_path.exists():
        return 0.0
    
    total_size = 0
    for f in cache_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
    
    return total_size / (1024 ** 3)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("SuperElite 区域检测测试")
    print("=" * 50)
    
    # 系统检测
    memory = get_system_memory_gb()
    print(f"\n系统内存: {memory:.1f} GB")
    
    passed, error = check_system_requirements()
    if passed:
        print("✅ 系统要求检测通过")
    else:
        print(f"❌ {error}")
    
    # 区域检测
    print("\n正在检测区域...")
    endpoint, name, is_china = get_recommended_endpoint()
    print(f"推荐下载源: {name}")
    print(f"是否中国大陆: {is_china}")
    
    # 模型缓存
    print(f"\n模型缓存路径: {get_model_cache_path()}")
    if is_model_cached():
        size = get_model_cache_size_gb()
        print(f"✅ 模型已缓存 ({size:.2f} GB)")
    else:
        print("❌ 模型未缓存")
