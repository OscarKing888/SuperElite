# -*- coding: utf-8 -*-
"""
SuperElite - 下载源选择对话框
首次运行时让用户选择模型下载源
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QRadioButton, QButtonGroup, QFrame
)
from PySide6.QtCore import Qt

import sys
from pathlib import Path

# 添加 backend 路径
backend_path = Path(__file__).parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from ui.styles import COLORS, FONTS


class DownloadSourceDialog(QDialog):
    """
    下载源选择对话框
    
    用户可以选择从官方源或国内镜像下载模型
    """
    
    ENDPOINT_OFFICIAL = "https://huggingface.co"
    ENDPOINT_MIRROR = "https://hf-mirror.com"
    
    def __init__(self, recommended_is_china: bool = False, parent=None):
        super().__init__(parent)
        self.recommended_is_china = recommended_is_china
        self.selected_endpoint = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle("下载 AI 模型")
        self.setFixedWidth(420)
        self.setModal(True)
        
        # 应用样式
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QRadioButton {{
                color: {COLORS['text_primary']};
                font-size: 14px;
                spacing: 8px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 标题
        title = QLabel("首次运行需要下载 AI 模型")
        title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(title)
        
        # 说明
        desc = QLabel("模型大小约 15GB，下载时间取决于网络速度。\n支持断点续传，可随时关闭程序后继续。")
        desc.setStyleSheet(f"""
            font-size: 13px;
            color: {COLORS['text_secondary']};
            line-height: 1.4;
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLORS['border']};")
        line.setFixedHeight(1)
        layout.addWidget(line)
        
        # 下载源选择
        source_label = QLabel("选择下载源:")
        source_label.setStyleSheet(f"font-size: 14px; font-weight: 500;")
        layout.addWidget(source_label)
        
        self.button_group = QButtonGroup(self)
        
        # 官方源
        self.radio_official = QRadioButton("官方源 (huggingface.co)")
        self.radio_official.setToolTip("适合海外用户或有代理的用户")
        self.button_group.addButton(self.radio_official)
        layout.addWidget(self.radio_official)
        
        # 国内镜像
        self.radio_mirror = QRadioButton("国内镜像 (hf-mirror.com)")
        self.radio_mirror.setToolTip("适合中国大陆用户，速度更快")
        self.button_group.addButton(self.radio_mirror)
        layout.addWidget(self.radio_mirror)
        
        # 根据推荐设置默认选中
        if self.recommended_is_china:
            self.radio_mirror.setChecked(True)
            tip = QLabel("💡 检测到您可能在中国大陆，已推荐使用国内镜像")
        else:
            self.radio_official.setChecked(True)
            tip = QLabel("💡 已推荐使用官方源")
        
        tip.setStyleSheet(f"""
            font-size: 12px;
            color: {COLORS['text_muted']};
            padding: 4px 0;
        """)
        layout.addWidget(tip)
        
        layout.addSpacing(8)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedWidth(100)
        cancel_btn.setObjectName("secondary")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        btn_layout.addStretch()
        
        start_btn = QPushButton("开始下载")
        start_btn.setFixedWidth(120)
        start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(start_btn)
        
        layout.addLayout(btn_layout)
    
    def _on_start(self):
        """开始下载"""
        if self.radio_mirror.isChecked():
            self.selected_endpoint = self.ENDPOINT_MIRROR
        else:
            self.selected_endpoint = self.ENDPOINT_OFFICIAL
        
        self.accept()
    
    def get_selected_endpoint(self) -> str:
        """获取用户选择的下载源"""
        return self.selected_endpoint


# 测试
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    
    # 模拟中国大陆用户
    dialog = DownloadSourceDialog(recommended_is_china=True)
    
    if dialog.exec():
        print(f"用户选择: {dialog.get_selected_endpoint()}")
    else:
        print("用户取消")
