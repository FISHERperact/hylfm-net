#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import argparse
import requests
import logging
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse, unquote
from PySide6.QtCore import QThread, Signal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("url_downloader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Downloader(QThread):
    """下载线程类，用于在后台下载文件并发送进度信号"""
    
    progress_signal = Signal(int, int)  # 已下载大小, 总大小
    finished = Signal()  # 完成信号
    error = Signal(str)  # 错误信号
    
    def __init__(self, url, save_path):
        """初始化下载器
        
        Args:
            url (str): 下载链接
            save_path (str): 保存路径
        """
        super().__init__()
        self.url = url
        self.save_path = save_path
        
    def run(self):
        """线程运行函数，执行下载任务"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
            
            # 发起请求
            print(f"正在连接到: {self.url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            print(f"开始下载文件到: {self.save_path}")
            print(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
            
            # 发送初始进度信号
            self.progress_signal.emit(0, total_size)
            
            with open(self.save_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if data:
                        f.write(data)
                        downloaded += len(data)
                        
                        # 发送进度信号 (每0.1秒更新一次)
                        self.progress_signal.emit(downloaded, total_size)
                        
                        # 让UI有时间处理信号
                        QThread.msleep(1)
            
            print(f"下载完成: {self.save_path}")
            self.finished.emit()
            
        except Exception as e:
            logging.error(f"下载出错: {str(e)}")
            self.error.emit(str(e))

def get_headers():
    """获取HTTP请求头"""
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
    }

def get_filename_from_url(url, response=None):
    """
    从URL或响应头中提取文件名
    
    参数:
        url (str): 文件URL
        response (Response, optional): 请求响应对象
    
    返回:
        str: 文件名
    """
    # 尝试从Content-Disposition获取文件名
    if response and response.headers.get('Content-Disposition'):
        filename_match = re.search(r'filename=["\']?([^"\';\n]+)', response.headers.get('Content-Disposition', ''))
        if filename_match:
            return filename_match.group(1)
    
    # 尝试从URL路径中提取文件名
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    filename = os.path.basename(path)
    
    # 如果URL中有文件名
    if filename and '.' in filename:
        return filename
    
    # 如果是特定网站，可以使用特定的提取方法
    if 'figshare.com' in url:
        url_parts = url.split('/')
        if len(url_parts) > 0:
            file_id = url_parts[-1].split('?')[0]
            if file_id.isdigit():
                return f"figshare_{file_id}"
    
    # 使用URL的哈希作为文件名
    return f"download_{abs(hash(url)) % 10000}"

def add_extension_based_on_content_type(filename, content_type):
    """
    根据Content-Type添加适当的文件扩展名
    
    参数:
        filename (str): 文件名
        content_type (str): Content-Type头
        
    返回:
        str: 带有适当扩展名的文件名
    """
    if '.' in filename:
        return filename
    
    # 常见MIME类型到扩展名的映射
    mime_to_ext = {
        'application/zip': '.zip',
        'application/x-zip-compressed': '.zip',
        'application/pdf': '.pdf',
        'application/octet-stream': '.bin',
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'text/plain': '.txt',
        'text/html': '.html',
        'text/csv': '.csv',
        'application/json': '.json',
        'application/xml': '.xml',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx'
    }
    
    # 查找适当的扩展名
    for mime, ext in mime_to_ext.items():
        if mime in content_type:
            return filename + ext
    
    # 如果没有找到匹配项，使用二进制扩展名
    return filename + '.bin'

def download_file(url, save_path, chunk_size=8192):
    """从URL下载文件（同步函数，用于非GUI环境）
    
    Args:
        url: 下载链接
        save_path: 保存路径
        chunk_size: 块大小
        
    Returns:
        bool: 下载是否成功
    """
    try:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # 获取文件名
        filename = os.path.basename(save_path)
        if not filename:
            filename = url.split('/')[-1]
            save_path = os.path.join(save_path, filename)
        
        # 发起请求
        print(f"正在连接到: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        print(f"开始下载 {filename} 到 {save_path}")
        print(f"文件大小: {total_size / (1024 * 1024):.2f} MB")
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                start_time = time.time()
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        
                        # 计算和显示下载速度
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            speed = downloaded / elapsed
                            pbar.set_postfix({
                                'speed': f"{speed / 1024:.1f} KB/s" if speed < 1024 * 1024 else f"{speed / (1024 * 1024):.1f} MB/s"
                            })
        
        print(f"下载完成: {save_path}")
        return True
        
    except Exception as e:
        logging.error(f"下载出错: {str(e)}")
        print(f"下载失败: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='从任意URL下载文件')
    parser.add_argument('urls', nargs='+', help='要下载的URL列表')
    parser.add_argument('--dir', type=str, default='downloads', help='文件保存目录')
    parser.add_argument('--retry', type=int, default=3, help='下载失败时的最大重试次数')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 下载所有URL
    success_count = 0
    total_count = len(args.urls)
    
    for idx, url in enumerate(args.urls, 1):
        logger.info(f"[{idx}/{total_count}] 处理URL: {url}")
        if download_file(url, save_dir):
            success_count += 1
    
    logger.info(f"下载完成: 成功 {success_count}/{total_count}")

if __name__ == "__main__":
    main()