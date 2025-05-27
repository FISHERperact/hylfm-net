#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import argparse
import datetime
import re
from pathlib import Path
import sys

# 设置控制台和文件系统编码
if sys.platform.startswith('win'):
    # 设置控制台编码为UTF-8
    import subprocess
    subprocess.run(["chcp", "65001"], shell=True, check=False)
    # 确保Python解释器使用UTF-8编码
    if hasattr(sys, 'setdefaultencoding'):
        sys.setdefaultencoding('utf-8')

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def extract_cell_number(filename):
    """从文件或文件夹名称中提取Cell编号"""
    match = re.search(r'Cell_(\d{3})', filename)
    if match:
        return int(match.group(1))
    return 0

def collect_microtubule_mrc_files(source_dir, target_dir, start_cell=1, end_cell=55, copy_method='copy', progress_callback=None):
    """
    按顺序收集微管数据集中的MRC文件
    
    参数:
    - source_dir: 微管数据集源目录路径
    - target_dir: 目标目录路径
    - start_cell: 起始Cell编号
    - end_cell: 结束Cell编号
    - copy_method: 复制方法，'copy'或'hardlink'
    - progress_callback: 进度回调函数，接收当前处理的Cell编号作为参数
    
    返回:
    - copied_count: 复制文件数量
    - skipped_count: 跳过文件数量
    - cell_processed: 处理的Cell数量
    - cell_skipped: 跳过的Cell数量
    """
    # 确保目标目录存在
    ensure_dir_exists(target_dir)
    
    copied_count = 0
    skipped_count = 0
    cell_processed = 0
    cell_skipped = 0
    
    # 处理指定范围内的每个Cell文件夹
    for cell_num in range(start_cell, end_cell + 1):
        # 更新进度
        if progress_callback:
            progress_callback(cell_num)
            
        cell_name = f"Cell_{cell_num:03d}"
        cell_dir = os.path.join(source_dir, cell_name)
        
        # 检查Cell文件夹是否存在
        if not os.path.exists(cell_dir):
            print(f"警告: {cell_dir} 不存在，跳过。")
            cell_skipped += 1
            continue
        
        print(f"\n处理 {cell_name}...")
        
        # 获取所有MRC文件
        mrc_files = glob.glob(os.path.join(cell_dir, "**/*.mrc"), recursive=True)
        
        if not mrc_files:
            print(f"警告: 在 {cell_dir} 中未找到MRC文件，跳过。")
            cell_skipped += 1
            continue
        
        cell_processed += 1
        
        # 处理每个MRC文件
        for mrc_file in mrc_files:
            # 构建目标文件路径
            rel_path = os.path.relpath(mrc_file, source_dir)
            target_file = os.path.join(target_dir, rel_path)
            target_file_dir = os.path.dirname(target_file)
            
            # 确保目标文件夹存在
            ensure_dir_exists(target_file_dir)
            
            # 如果目标文件已存在，跳过
            if os.path.exists(target_file):
                print(f"文件已存在，跳过: {rel_path}")
                skipped_count += 1
                continue
            
            try:
                # 根据指定方法复制文件
                if copy_method == 'hardlink':
                    os.link(mrc_file, target_file)
                else:
                    shutil.copy2(mrc_file, target_file)
                print(f"已复制: {rel_path}")
                copied_count += 1
            except Exception as e:
                print(f"复制文件时出错: {rel_path}")
                print(f"错误信息: {str(e)}")
                skipped_count += 1
                
    print(f"\n处理完成:")
    print(f"- 已复制: {copied_count} 个文件")
    print(f"- 已跳过: {skipped_count} 个文件")
    print(f"- 已处理: {cell_processed} 个Cell")
    print(f"- 已跳过: {cell_skipped} 个Cell")
    
    return copied_count, skipped_count, cell_processed, cell_skipped

def main():
    parser = argparse.ArgumentParser(description='按顺序收集微管数据集中的MRC文件')
    
    # 参数设置
    parser.add_argument('--source-dir', '-s', type=str, 
                       default=r'D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\Mic483生成数据集（1.0）',
                       help='微管数据集源目录路径')
    parser.add_argument('--target-dir', '-t', type=str, 
                       default='收集的微管MRC文件',
                       help='目标目录路径')
    parser.add_argument('--start-cell', '-sc', type=int, default=1,
                       help='起始Cell编号')
    parser.add_argument('--end-cell', '-ec', type=int, default=55,
                       help='结束Cell编号')
    parser.add_argument('--hardlink', '-l', action='store_true',
                       help='使用硬链接代替复制以节省磁盘空间')
    
    args = parser.parse_args()
    
    # 如果未指定目标目录，添加日期后缀
    if args.target_dir == '收集的微管MRC文件':
        date_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.target_dir = f"{args.target_dir}_{date_suffix}"
    
    # 收集文件
    print(f"从 {args.source_dir} 收集MRC文件到 {args.target_dir}...")
    copy_method = 'hardlink' if args.hardlink else 'copy'
    
    copied_count, skipped_count, cell_processed, cell_skipped = collect_microtubule_mrc_files(
        args.source_dir, 
        args.target_dir, 
        args.start_cell,
        args.end_cell,
        copy_method
    )
    
    # 打印结果
    print(f"\n操作完成!")
    print(f"处理的Cell: {cell_processed} 个")
    print(f"跳过的Cell: {cell_skipped} 个")
    print(f"成功复制: {copied_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件 (已存在)")
    print(f"所有文件已复制到: {os.path.abspath(args.target_dir)}")

if __name__ == "__main__":
    main() 