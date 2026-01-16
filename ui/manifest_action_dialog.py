# -*- coding: utf-8 -*-
"""
SuperElite - Manifest 操作对话框
用于已处理目录的操作选择
"""

import sys
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QGridLayout
)
from PySide6.QtCore import Qt

# 添加 backend 路径
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from ui.styles import COLORS, FONTS, GLOBAL_STYLE


class ManifestActionDialog(QDialog):
    """
    已处理目录操作对话框
    
    当检测到目录已完成处理时弹出，提供三个选项:
    - 重新评星 (使用缓存分数)
    - 重置数据 (清除所有)
    - 取消
    """
    
    # 返回值常量
    ACTION_CANCEL = 0
    ACTION_RERATE = 1
    ACTION_RESET = 2
    ACTION_CONTINUE = 3  # 继续未完成的处理
    
    def __init__(self, parent=None, summary: dict = None, is_in_progress: bool = False, 
                 current_thresholds: tuple = None):
        """
        初始化
        
        Args:
            parent: 父窗口
            summary: manifest 摘要信息
            is_in_progress: 是否是未完成的处理
            current_thresholds: 当前主界面选择的阈值（如果是自定义的）
        """
        super().__init__(parent)
        
        self.summary = summary or {}
        self.is_in_progress = is_in_progress
        self.result_action = self.ACTION_CANCEL
        self.current_thresholds = current_thresholds or (78.0, 72.0, 66.0, 58.0)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle("检测到历史处理记录")
        self.setMinimumWidth(420)
        self.setStyleSheet(GLOBAL_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # 标题
        if self.is_in_progress:
            title = QLabel("⚠️ 检测到未完成的处理任务")
            title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['warning']};")
        else:
            title = QLabel("📋 检测到该目录已完成评分")
            title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text_primary']};")
        layout.addWidget(title)
        
        # 信息面板
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 10px;
                padding: 16px;
            }}
        """)
        info_layout = QGridLayout(info_frame)
        info_layout.setSpacing(8)
        
        # 处理时间
        created_at = self.summary.get("created_at", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
                created_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                created_str = created_at[:16]
        else:
            created_str = "-"
        
        info_layout.addWidget(self._label("处理时间:"), 0, 0)
        info_layout.addWidget(self._value(created_str), 0, 1)
        
        # 文件数量
        total = self.summary.get("total_files", 0)
        processed = self.summary.get("processed_files", 0)
        
        if self.is_in_progress:
            files_str = f"{processed} / {total} 张"
        else:
            files_str = f"{total} 张"
        
        info_layout.addWidget(self._label("文件数量:"), 1, 0)
        info_layout.addWidget(self._value(files_str), 1, 1)
        
        # 当前阈值
        thresholds = self.summary.get("thresholds", [78, 72, 66, 58])
        thresh_str = " / ".join(str(int(t)) for t in thresholds)
        
        info_layout.addWidget(self._label("当前阈值:"), 2, 0)
        info_layout.addWidget(self._value(thresh_str), 2, 1)
        
        # 星级分布 (如果已完成)
        if not self.is_in_progress:
            by_rating = self.summary.get("by_rating", {})
            dist_parts = []
            for star in [4, 3, 2, 1, 0]:
                count = by_rating.get(star, 0)
                if count > 0:
                    dist_parts.append(f"{star}★:{count}")
            dist_str = "  ".join(dist_parts) if dist_parts else "-"
            
            info_layout.addWidget(self._label("星级分布:"), 3, 0)
            info_layout.addWidget(self._value(dist_str), 3, 1)
        
        layout.addWidget(info_frame)
        
        # 新阈值选择器 (仅在已完成时显示)
        if not self.is_in_progress:
            from PySide6.QtWidgets import QComboBox
            
            preset_frame = QFrame()
            preset_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['bg_elevated']};
                    border-radius: 10px;
                    padding: 12px 16px;
                }}
            """)
            preset_layout = QHBoxLayout(preset_frame)
            preset_layout.setContentsMargins(0, 0, 0, 0)
            
            preset_label = QLabel("选择评分标准:")
            preset_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
            preset_layout.addWidget(preset_label)
            
            self.preset_combo = QComboBox()
            
            # 当前阈值作为第一选项
            t = self.current_thresholds
            current_str = f"当前 ({t[0]:.0f} / {t[1]:.0f} / {t[2]:.0f} / {t[3]:.0f})"
            
            self.preset_combo.addItems([
                current_str,
                "默认 (78 / 72 / 66 / 58)",
                "严格 (85 / 80 / 75 / 70)",
                "宽松 (70 / 60 / 50 / 40)",
            ])
            self.preset_combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {COLORS['bg_secondary']};
                    color: {COLORS['text_primary']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 6px;
                    padding: 6px 12px;
                    min-width: 180px;
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {COLORS['bg_elevated']};
                    color: {COLORS['text_primary']};
                    selection-background-color: {COLORS['accent']};
                }}
            """)
            preset_layout.addWidget(self.preset_combo, 1)
            
            layout.addWidget(preset_frame)
        
        # 提示文字
        if self.is_in_progress:
            hint = QLabel("上次处理未完成，您可以继续处理或重新开始。")
        else:
            hint = QLabel("您可以使用新的阈值重新评星，或重置所有数据重新处理。")
        hint.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 12px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("tertiary")
        cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(cancel_btn)
        
        btn_layout.addStretch()
        
        # 重置按钮
        reset_btn = QPushButton("重置数据")
        reset_btn.setObjectName("secondary")
        reset_btn.setToolTip("清除所有评分数据，重新开始处理")
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)
        
        # 主按钮
        if self.is_in_progress:
            main_btn = QPushButton("继续处理")
            main_btn.setToolTip("继续处理剩余的文件")
            main_btn.clicked.connect(self._on_continue)
        else:
            main_btn = QPushButton("重新评星")
            main_btn.setToolTip("使用当前阈值重新计算星级（不重跑AI）")
            main_btn.clicked.connect(self._on_rerate)
        
        btn_layout.addWidget(main_btn)
        
        layout.addLayout(btn_layout)
    
    def _label(self, text: str) -> QLabel:
        """创建标签"""
        label = QLabel(text)
        label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 13px;")
        return label
    
    def _value(self, text: str) -> QLabel:
        """创建值"""
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: 500;
            font-family: {FONTS['mono']};
        """)
        return label
    
    def _on_cancel(self):
        """取消"""
        self.result_action = self.ACTION_CANCEL
        self.reject()
    
    def _on_rerate(self):
        """重新评星"""
        self.result_action = self.ACTION_RERATE
        self.accept()
    
    def _on_reset(self):
        """重置数据"""
        self.result_action = self.ACTION_RESET
        self.accept()
    
    def _on_continue(self):
        """继续处理"""
        self.result_action = self.ACTION_CONTINUE
        self.accept()
    
    def get_action(self) -> int:
        """获取用户选择的操作"""
        return self.result_action
    
    def get_selected_thresholds(self) -> tuple:
        """获取用户选择的阈值"""
        # 预设阈值映射 (0=当前, 1=默认, 2=严格, 3=宽松)
        presets = {
            0: self.current_thresholds,       # 当前
            1: (78.0, 72.0, 66.0, 58.0),       # 默认
            2: (85.0, 80.0, 75.0, 70.0),       # 严格
            3: (70.0, 60.0, 50.0, 40.0),       # 宽松
        }
        
        if hasattr(self, 'preset_combo'):
            index = self.preset_combo.currentIndex()
            return presets.get(index, self.current_thresholds)
        
        return self.current_thresholds


# 测试代码
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    
    # 模拟数据
    summary = {
        "status": "completed",
        "created_at": "2026-01-15T14:30:00",
        "updated_at": "2026-01-15T14:45:00",
        "total_files": 128,
        "processed_files": 128,
        "thresholds": [78, 72, 66, 58],
        "by_rating": {4: 15, 3: 28, 2: 45, 1: 25, 0: 15},
    }
    
    dialog = ManifestActionDialog(summary=summary, is_in_progress=False)
    result = dialog.exec()
    
    if result:
        action = dialog.get_action()
        print(f"用户选择: {action}")
    else:
        print("用户取消")
