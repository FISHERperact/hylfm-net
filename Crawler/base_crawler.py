import requests
import time
import logging
import os
from tqdm import tqdm

class BaseDownloader:
    """基础下载器类，提供基本的下载功能和进度回调"""
    
    def __init__(self):
        """初始化下载器"""
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """设置进度回调函数
        
        Args:
            callback: 回调函数，接收三个参数：已下载大小(MB)、总大小(MB)、下载速度(KB/s)
        """
        self.progress_callback = callback
        
    def update_progress(self, downloaded_bytes, total_bytes, speed_bytes):
        """更新下载进度
        
        Args:
            downloaded_bytes: 已下载的字节数
            total_bytes: 总字节数
            speed_bytes: 下载速度(字节/秒)
        """
        if self.progress_callback:
            downloaded_mb = downloaded_bytes / (1024 * 1024)  # 转换为MB
            total_mb = total_bytes / (1024 * 1024)  # 转换为MB
            speed_kb = speed_bytes / 1024  # 转换为KB/s
            self.progress_callback(downloaded_mb, total_mb, speed_kb)
    
    def download(self, url, save_path):
        """下载文件的基本实现
        
        Args:
            url: 下载链接
            save_path: 保存路径
            
        Returns:
            bool: 下载是否成功
            
        Raises:
            Exception: 下载过程中的任何错误
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 发起请求
            print(f"正在连接到: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            start_time = time.time()
            
            print(f"开始下载文件到: {save_path}")
            print(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
            
            with open(save_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if data:
                        downloaded += len(data)
                        f.write(data)
                        
                        # 计算下载速度
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            
                            # 更新进度
                            self.update_progress(downloaded, total_size, speed)
            
            print(f"下载完成: {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"下载出错: {str(e)}")
            raise 