# -*- coding: utf-8 -*-
"""
SuperElite API Server - 深度评片引擎
内嵌在 GUI 中的 HTTP 服务，供 Lightroom Plugin 调用
"""

import socket
import threading
from flask import Flask, jsonify, request
import logging

# 禁用 Flask 的默认日志（太吵了）
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class APIServer:
    """轻量级 HTTP API 服务器"""
    
    DEFAULT_PORT = 52765  # SuperElite 专用端口
    
    def __init__(self, port: int = None):
        self.port = port or self.DEFAULT_PORT
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = False
        self._setup_routes()
    
    def _setup_routes(self):
        """设置 API 路由"""
        
        @self.app.route('/status', methods=['GET'])
        def status():
            """健康检查和状态查询"""
            return jsonify({
                "status": "running",
                "service": "SuperElite Deep Analysis Engine",
                "version": "1.0",
                "port": self.port,
                "model": "co-instruct",
                "model_loaded": False,  # TODO: 实际检查模型状态
            })
        
        @self.app.route('/ping', methods=['GET'])
        def ping():
            """简单的 ping 测试"""
            return jsonify({"pong": True})
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze():
            """分析图片 - 调用 Co-Instruct 模型"""
            data = request.get_json() or {}
            image_path = data.get('image') or data.get('image_path')
            language = data.get('language', 'cn')  # 默认中文
            tasks = data.get('tasks', ['keywords', 'caption', 'title', 'scene', 'mood'])
            
            if not image_path:
                return jsonify({"error": "Missing 'image' parameter", "success": False}), 400
            
            # 检查文件是否存在
            import os
            if not os.path.exists(image_path):
                return jsonify({"error": f"File not found: {image_path}", "success": False}), 404
            
            try:
                from coinstruct_analyzer import analyze as coinstruct_analyze
                result = coinstruct_analyze(image_path, tasks=tasks, language=language)
                return jsonify(result)
            except Exception as e:
                return jsonify({
                    "error": str(e),
                    "success": False
                }), 500
    
    @staticmethod
    def is_port_available(port: int) -> bool:
        """检查端口是否可用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False
    
    def start(self) -> tuple[bool, str]:
        """
        启动服务器
        
        Returns:
            (success, message)
        """
        if self.running:
            return True, "服务已在运行"
        
        # 检查端口
        if not self.is_port_available(self.port):
            return False, f"端口 {self.port} 已被占用"
        
        try:
            self.running = True
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True,
                name="SuperEliteAPIServer"
            )
            self.server_thread.start()
            return True, f"服务已启动: http://127.0.0.1:{self.port}"
        except Exception as e:
            self.running = False
            return False, f"启动失败: {e}"
    
    def _run_server(self):
        """在后台线程运行 Flask 服务"""
        try:
            # 使用 werkzeug 的 serving 模块
            from werkzeug.serving import make_server
            self.server = make_server('127.0.0.1', self.port, self.app, threaded=True)
            self.server.serve_forever()
        except Exception as e:
            print(f"API Server error: {e}")
            self.running = False
    
    def stop(self):
        """停止服务器"""
        if not self.running:
            return
        
        self.running = False
        
        # 关闭 werkzeug 服务器
        if hasattr(self, 'server'):
            self.server.shutdown()
        
        # 等待线程结束
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
    
    @property
    def is_running(self) -> bool:
        """服务是否正在运行"""
        return self.running


# 单例模式
_api_server_instance = None

def get_api_server(port: int = None) -> APIServer:
    """获取 API 服务器单例"""
    global _api_server_instance
    if _api_server_instance is None:
        _api_server_instance = APIServer(port)
    return _api_server_instance


# 测试
if __name__ == "__main__":
    print("🚀 启动 SuperElite API Server...")
    server = get_api_server()
    success, msg = server.start()
    print(f"   {msg}")
    
    if success:
        print(f"\n📡 测试端点:")
        print(f"   curl http://127.0.0.1:{server.port}/status")
        print(f"   curl http://127.0.0.1:{server.port}/ping")
        print(f"\n按 Ctrl+C 停止...")
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n停止服务...")
            server.stop()
            print("✅ 已停止")
