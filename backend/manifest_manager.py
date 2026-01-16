# -*- coding: utf-8 -*-
"""
SuperElite - Manifest 管理器
用于记录处理状态，支持增量处理、中断恢复、快速重评星
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Manifest 文件名
MANIFEST_FILENAME = ".superelite_manifest.json"
MANIFEST_VERSION = "1.0"


class ManifestManager:
    """Manifest 管理器 - 记录目录处理状态"""
    
    def __init__(self, directory: str):
        """
        初始化
        
        Args:
            directory: 目标目录路径
        """
        self.directory = Path(directory)
        self.manifest_path = self.directory / MANIFEST_FILENAME
        self.data: Dict[str, Any] = {}
        
        # 如果存在 manifest，加载它
        if self.manifest_path.exists():
            self.load()
        else:
            self._init_new_manifest()
    
    def _init_new_manifest(self):
        """初始化新的 manifest"""
        self.data = {
            "version": MANIFEST_VERSION,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": {
                "thresholds": [78.0, 72.0, 66.0, 58.0],
                "quality_weight": 0.4,
                "aesthetic_weight": 0.6,
            },
            "status": "new",  # new / in_progress / completed
            "total_files": 0,
            "processed_files": 0,
            "files": {},
        }
    
    def load(self) -> bool:
        """
        加载 manifest 文件
        
        Returns:
            True 如果加载成功
        """
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            
            # 兼容旧版 manifest (files 是列表而非字典)
            self._migrate_old_format()
            
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Manifest] 加载失败: {e}")
            self._init_new_manifest()
            return False
    
    def _migrate_old_format(self):
        """迁移旧版 manifest 格式"""
        files_data = self.data.get("files", {})
        
        # 检测旧格式: files 是列表
        if isinstance(files_data, list):
            print("[Manifest] 检测到旧版格式，正在迁移...")
            new_files = {}
            
            for file_info in files_data:
                filename = file_info.get("filename", "")
                if filename:
                    scores = file_info.get("scores", {})
                    new_files[filename] = {
                        "hash": "",  # 旧格式没有 hash
                        "quality": scores.get("quality", 0),
                        "aesthetic": scores.get("aesthetic", 0),
                        "total": scores.get("total", 0),
                        "rating": file_info.get("rating", 0),
                        "processed_at": self.data.get("created", ""),
                    }
            
            self.data["files"] = new_files
            self.data["processed_files"] = len(new_files)
            
            # 更新版本信息
            if "version" not in self.data:
                self.data["version"] = MANIFEST_VERSION
            if "status" not in self.data:
                self.data["status"] = "completed"  # 旧版没有状态，假设已完成
            if "created_at" not in self.data:
                self.data["created_at"] = self.data.get("created", "")
            if "updated_at" not in self.data:
                self.data["updated_at"] = self.data.get("created", "")
            if "config" not in self.data:
                settings = self.data.get("settings", {})
                self.data["config"] = {
                    "thresholds": settings.get("thresholds", [78, 72, 66, 58]),
                    "quality_weight": settings.get("quality_weight", 0.4),
                    "aesthetic_weight": settings.get("aesthetic_weight", 0.6),
                }
            if "total_files" not in self.data:
                stats = self.data.get("statistics", {})
                self.data["total_files"] = stats.get("total", len(new_files))
            
            # 保存迁移后的格式
            self.save()
            print(f"[Manifest] 迁移完成: {len(new_files)} 个文件")
    
    def save(self):
        """保存 manifest 到文件"""
        self.data["updated_at"] = datetime.now().isoformat()
        
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[Manifest] 保存失败: {e}")
    
    def delete(self):
        """删除 manifest 文件"""
        if self.manifest_path.exists():
            os.remove(self.manifest_path)
            self._init_new_manifest()
    
    def restore_files(self) -> Dict[str, int]:
        """
        将文件从星级子目录恢复到顶层目录
        
        Returns:
            {"moved": int, "failed": int, "already_in_place": int}
        """
        import shutil
        
        result = {"moved": 0, "failed": 0, "already_in_place": 0}
        
        # 星级目录名
        star_dirs = ["0星", "1星", "2星", "3星", "4星", "5星"]
        
        # 遍历所有星级子目录
        for star_dir_name in star_dirs:
            star_dir = self.directory / star_dir_name
            
            if not star_dir.exists() or not star_dir.is_dir():
                continue
            
            # 遍历该目录下的所有文件
            for file_path in star_dir.iterdir():
                if not file_path.is_file():
                    continue
                
                # 目标位置：顶层目录
                dest_path = self.directory / file_path.name
                
                if dest_path.exists():
                    # 文件已存在于顶层，可能是重复
                    result["already_in_place"] += 1
                    continue
                
                try:
                    shutil.move(str(file_path), str(dest_path))
                    result["moved"] += 1
                except Exception as e:
                    print(f"[Manifest] 移动失败 {file_path.name}: {e}")
                    result["failed"] += 1
        
        # 删除空的星级目录
        for star_dir_name in star_dirs:
            star_dir = self.directory / star_dir_name
            if star_dir.exists() and star_dir.is_dir():
                try:
                    # 只删除空目录
                    if not any(star_dir.iterdir()):
                        star_dir.rmdir()
                except Exception:
                    pass
        
        return result
    
    # ==================== 状态查询 ====================
    
    @property
    def status(self) -> str:
        """获取处理状态"""
        return self.data.get("status", "new")
    
    @status.setter
    def status(self, value: str):
        """设置处理状态"""
        self.data["status"] = value
    
    @property
    def is_completed(self) -> bool:
        """是否已完成处理"""
        return self.status == "completed"
    
    @property
    def is_in_progress(self) -> bool:
        """是否处理中（被中断）"""
        return self.status == "in_progress"
    
    @property
    def processed_count(self) -> int:
        """已处理文件数"""
        return self.data.get("processed_files", 0)
    
    @property
    def total_count(self) -> int:
        """总文件数"""
        return self.data.get("total_files", 0)
    
    @property
    def config(self) -> Dict:
        """获取处理配置"""
        return self.data.get("config", {})
    
    @property
    def created_at(self) -> str:
        """创建时间"""
        return self.data.get("created_at", "")
    
    @property
    def updated_at(self) -> str:
        """最后更新时间"""
        return self.data.get("updated_at", "")
    
    # ==================== 文件操作 ====================
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        计算文件 MD5 hash (只读前 64KB，速度快)
        
        Args:
            file_path: 文件路径
            
        Returns:
            MD5 hash 字符串
        """
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # 只读前 64KB 作为指纹，避免大文件太慢
                chunk = f.read(65536)
                hasher.update(chunk)
                # 再读文件末尾 64KB
                f.seek(-min(65536, os.path.getsize(file_path)), 2)
                chunk = f.read(65536)
                hasher.update(chunk)
        except IOError:
            return ""
        return hasher.hexdigest()
    
    def is_file_processed(self, filename: str, file_path: str) -> bool:
        """
        检查文件是否已处理且未修改
        
        Args:
            filename: 文件名
            file_path: 文件完整路径
            
        Returns:
            True 如果已处理且文件未修改
        """
        if filename not in self.data.get("files", {}):
            return False
        
        file_info = self.data["files"][filename]
        stored_hash = file_info.get("hash", "")
        
        if not stored_hash:
            return False
        
        # 计算当前文件 hash
        current_hash = self.calculate_file_hash(file_path)
        
        return stored_hash == current_hash
    
    def get_file_scores(self, filename: str) -> Optional[Dict]:
        """
        获取文件的缓存分数
        
        Args:
            filename: 文件名
            
        Returns:
            {"quality": float, "aesthetic": float, "total": float, "rating": int}
            或 None 如果不存在
        """
        if filename not in self.data.get("files", {}):
            return None
        
        file_info = self.data["files"][filename]
        return {
            "quality": file_info.get("quality", 0),
            "aesthetic": file_info.get("aesthetic", 0),
            "total": file_info.get("total", 0),
            "rating": file_info.get("rating", 0),
        }
    
    def add_file_result(
        self,
        filename: str,
        file_path: str,
        quality: float,
        aesthetic: float,
        total: float,
        rating: int,
    ):
        """
        添加/更新文件处理结果
        
        Args:
            filename: 文件名
            file_path: 文件完整路径
            quality: 质量分
            aesthetic: 美学分
            total: 综合分
            rating: 星级
        """
        file_hash = self.calculate_file_hash(file_path)
        
        self.data["files"][filename] = {
            "hash": file_hash,
            "original_path": file_path,  # 记录原始路径，用于恢复
            "quality": round(quality, 2),
            "aesthetic": round(aesthetic, 2),
            "total": round(total, 2),
            "rating": rating,
            "processed_at": datetime.now().isoformat(),
        }
        
        self.data["processed_files"] = len(self.data["files"])
    
    def update_file_rating(self, filename: str, new_rating: int):
        """
        只更新文件的星级（用于快速重评星）
        
        Args:
            filename: 文件名
            new_rating: 新星级
        """
        if filename in self.data.get("files", {}):
            self.data["files"][filename]["rating"] = new_rating
            self.data["files"][filename]["processed_at"] = datetime.now().isoformat()
    
    # ==================== 批量操作 ====================
    
    def set_config(
        self,
        thresholds: Tuple[float, ...],
        quality_weight: float,
        aesthetic_weight: float,
    ):
        """设置处理配置"""
        self.data["config"] = {
            "thresholds": list(thresholds),
            "quality_weight": quality_weight,
            "aesthetic_weight": aesthetic_weight,
        }
    
    def set_total_files(self, count: int):
        """设置总文件数"""
        self.data["total_files"] = count
    
    def start_processing(self):
        """标记开始处理"""
        self.status = "in_progress"
        self.save()
    
    def complete_processing(self):
        """标记处理完成"""
        self.status = "completed"
        self.save()
    
    def get_pending_files(self, all_files: List[str]) -> List[str]:
        """
        获取待处理文件列表（跳过已处理的）
        
        Args:
            all_files: 目录中所有文件路径列表
            
        Returns:
            需要处理的文件路径列表
        """
        pending = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            if not self.is_file_processed(filename, file_path):
                pending.append(file_path)
        
        return pending
    
    def get_all_cached_scores(self) -> Dict[str, Dict]:
        """
        获取所有缓存的分数
        
        Returns:
            {filename: {quality, aesthetic, total, rating}}
        """
        return {
            filename: {
                "quality": info.get("quality", 0),
                "aesthetic": info.get("aesthetic", 0),
                "total": info.get("total", 0),
                "rating": info.get("rating", 0),
            }
            for filename, info in self.data.get("files", {}).items()
        }
    
    def get_summary(self) -> Dict:
        """
        获取处理摘要信息
        
        Returns:
            {
                "status": str,
                "created_at": str,
                "updated_at": str,
                "total_files": int,
                "processed_files": int,
                "thresholds": list,
                "by_rating": {rating: count}
            }
        """
        # 统计各星级数量
        by_rating = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for file_info in self.data.get("files", {}).values():
            rating = file_info.get("rating", 0)
            by_rating[rating] = by_rating.get(rating, 0) + 1
        
        return {
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_files": self.total_count,
            "processed_files": self.processed_count,
            "thresholds": self.config.get("thresholds", []),
            "by_rating": by_rating,
        }


# ==================== 便捷函数 ====================

def get_manifest(directory: str) -> ManifestManager:
    """获取目录的 manifest 管理器"""
    return ManifestManager(directory)


def has_manifest(directory: str) -> bool:
    """检查目录是否有 manifest 文件"""
    manifest_path = Path(directory) / MANIFEST_FILENAME
    return manifest_path.exists()


def quick_rerate(
    directory: str,
    new_thresholds: Tuple[float, float, float, float],
    quality_weight: float = 0.4,
    aesthetic_weight: float = 0.6,
) -> List[Dict]:
    """
    快速重评星 - 使用缓存分数重新计算星级
    
    Args:
        directory: 目录路径
        new_thresholds: 新阈值 (4星, 3星, 2星, 1星)
        quality_weight: 质量权重
        aesthetic_weight: 美学权重
        
    Returns:
        更新的文件列表 [{filename, old_rating, new_rating, total}]
    """
    manager = ManifestManager(directory)
    
    if not manager.is_completed:
        raise ValueError("目录未完成处理，无法快速重评星")
    
    t4, t3, t2, t1 = new_thresholds
    results = []
    
    for filename, scores in manager.get_all_cached_scores().items():
        old_rating = scores["rating"]
        
        # 重新计算综合分
        total = (
            scores["quality"] * quality_weight +
            scores["aesthetic"] * aesthetic_weight
        )
        
        # 新的星级映射
        if total >= t4:
            new_rating = 4
        elif total >= t3:
            new_rating = 3
        elif total >= t2:
            new_rating = 2
        elif total >= t1:
            new_rating = 1
        else:
            new_rating = 0
        
        # 更新 manifest
        manager.update_file_rating(filename, new_rating)
        
        results.append({
            "filename": filename,
            "old_rating": old_rating,
            "new_rating": new_rating,
            "total": round(total, 2),
            "changed": old_rating != new_rating,
        })
    
    # 更新配置并保存
    manager.set_config(new_thresholds, quality_weight, aesthetic_weight)
    manager.save()
    
    return results


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python manifest_manager.py <目录>")
        sys.exit(1)
    
    directory = sys.argv[1]
    manager = ManifestManager(directory)
    
    print(f"\n{'='*50}")
    print(f"目录: {directory}")
    print(f"{'='*50}")
    
    summary = manager.get_summary()
    print(f"状态: {summary['status']}")
    print(f"创建时间: {summary['created_at']}")
    print(f"更新时间: {summary['updated_at']}")
    print(f"总文件数: {summary['total_files']}")
    print(f"已处理: {summary['processed_files']}")
    print(f"阈值: {summary['thresholds']}")
    print(f"\n星级分布:")
    for rating in range(5, -1, -1):
        count = summary['by_rating'].get(rating, 0)
        print(f"  {'★' * rating}{'☆' * (5-rating)}: {count}")
