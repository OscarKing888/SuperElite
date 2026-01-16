# -*- coding: utf-8 -*-
"""
SuperElite - 模型下载器
后台下载 HuggingFace 模型，支持进度显示和断点续传
"""

import os
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

# 添加 backend 路径
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


class ModelDownloader(QThread):
    """
    后台模型下载线程
    
    Signals:
        progress(int, str): (百分比, 状态描述)
        log_message(str, str): (类型, 消息) - 用于在 GUI 控制台输出
        finished(bool, str): (成功, 消息)
    """
    
    progress = Signal(int, str)
    log_message = Signal(str, str)  # (type, message)
    finished = Signal(bool, str)
    
    MODEL_ID = "q-future/one-align"
    
    def __init__(self, endpoint: str = "https://huggingface.co", parent=None):
        super().__init__(parent)
        self.endpoint = endpoint
        self._should_stop = False
    
    def run(self):
        """执行下载"""
        try:
            from region_detector import setup_hf_endpoint
            
            # 设置下载源
            setup_hf_endpoint(self.endpoint)
            
            self.progress.emit(0, "正在连接服务器...")
            self.log_message.emit("info", "📡 正在连接下载服务器...")
            
            from huggingface_hub import snapshot_download, HfFileSystem
            from huggingface_hub.utils import tqdm as hf_tqdm
            
            # 获取模型文件列表
            self.log_message.emit("info", "📋 获取模型文件列表...")
            self.progress.emit(5, "获取文件列表...")
            
            try:
                fs = HfFileSystem()
                files = fs.ls(f"{self.MODEL_ID}", detail=True)
                total_size = sum(f.get('size', 0) for f in files if f.get('size'))
                total_size_gb = total_size / (1024**3)
                self.log_message.emit("info", f"   模型大小: {total_size_gb:.2f} GB")
                self.log_message.emit("info", f"   文件数量: {len(files)} 个")
            except Exception:
                self.log_message.emit("warning", "   ⚠️ 无法获取文件列表，继续下载...")
            
            self.log_message.emit("info", "")
            self.log_message.emit("info", "⬇️ 开始下载模型文件...")
            self.log_message.emit("default", "   (HuggingFace 会显示单个文件进度)")
            self.log_message.emit("info", "")
            self.progress.emit(10, "下载中...")
            
            # 使用 snapshot_download 下载模型
            # 原生支持断点续传
            # 进度会在终端显示，我们捕获并转发到 GUI
            
            import io
            import contextlib
            
            # 自定义进度处理
            class ProgressCapture:
                def __init__(self, downloader):
                    self.downloader = downloader
                    self.last_file = ""
                    self.file_count = 0
                
                def __call__(self, *args, **kwargs):
                    # 这是 tqdm 的进度回调
                    desc = kwargs.get('desc', '')
                    if desc and desc != self.last_file:
                        self.last_file = desc
                        self.file_count += 1
                        # 简化文件名
                        short_name = desc.split('/')[-1] if '/' in desc else desc
                        if len(short_name) > 35:
                            short_name = f"...{short_name[-32:]}"
                        self.downloader.log_message.emit("default", f"   📄 [{self.file_count}] {short_name}")
                        # 估算进度 (10-95%)
                        progress = min(10 + self.file_count * 5, 95)
                        self.downloader.progress.emit(progress, short_name)
                    return hf_tqdm(*args, **kwargs)
            
            snapshot_download(
                repo_id=self.MODEL_ID,
                # 不使用自定义 tqdm，让终端显示原生进度
            )
            
            self.progress.emit(100, "下载完成")
            self.log_message.emit("info", "")
            self.log_message.emit("success", "✅ 模型下载完成！")
            self.finished.emit(True, "模型下载成功")
            
        except Exception as e:
            error_msg = str(e)
            if "ConnectTimeout" in error_msg or "Connection" in error_msg:
                error_msg = "网络连接失败，请检查网络或更换下载源"
            elif "403" in error_msg:
                error_msg = "访问被拒绝，请尝试更换下载源"
            self.log_message.emit("error", f"❌ 下载失败: {error_msg}")
            self.finished.emit(False, error_msg)
    
    def stop(self):
        """请求停止"""
        self._should_stop = True


# 测试
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    
    def on_progress(percent, desc):
        print(f"[{percent:3d}%] {desc}")
    
    def on_log(log_type, msg):
        print(f"  [{log_type}] {msg}")
    
    def on_finished(success, msg):
        print(f"\n{'✅' if success else '❌'} {msg}")
        app.quit()
    
    downloader = ModelDownloader()
    downloader.progress.connect(on_progress)
    downloader.log_message.connect(on_log)
    downloader.finished.connect(on_finished)
    
    print("开始下载测试...")
    downloader.start()
    
    app.exec()
