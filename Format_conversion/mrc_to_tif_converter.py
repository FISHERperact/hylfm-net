#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRC文件转换为TIF文件工具
"""

import os
import numpy as np
import tifffile
import mrcfile
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MRC转TIF")

def mrc_to_tif(input_file, output_file):
    """
    将MRC文件转换为TIF文件
    
    参数:
        input_file: 输入MRC文件路径
        output_file: 输出TIF文件路径
    """
    try:
        logger.info(f"处理文件: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 打开MRC文件
        with mrcfile.open(input_file, permissive=True) as mrc:
            # 获取数据
            data = mrc.data
            
            # 数据类型兼容性检查
            if data.dtype != np.float32 and data.dtype != np.int16 and data.dtype != np.uint16:
                logger.warning(f"转换数据类型 {data.dtype} 到 float32")
                data = data.astype(np.float32)
            
            # 检查数据是否有NaN或Inf
            if np.isnan(data).any() or np.isinf(data).any():
                logger.warning(f"数据中包含NaN或Inf值，将替换为0")
                data = np.nan_to_num(data, nan=0.0, posinf=65535.0, neginf=0.0)
            
            # 检查维度
            if len(data.shape) < 2:
                raise ValueError(f"数据维度过低: {data.shape}")
            
            # 规范化数据到uint16范围 (适合于TIF)
            min_val = np.min(data)
            max_val = np.max(data)
            
            if max_val > min_val:
                data_normalized = ((data - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
            else:
                # 如果数据是常数，设置为中间值
                data_normalized = np.ones_like(data, dtype=np.uint16) * 32767
            
            # 将数据保存为TIF文件
            tifffile.imwrite(output_file, data_normalized)
            
            logger.info(f"转换成功: {os.path.basename(output_file)}")
            
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        raise

def batch_convert_mrc_to_tif(input_dir, output_dir):
    """
    批量将目录中的MRC文件转换为TIF文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
    
    返回:
        成功转换的文件列表和失败的文件列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有MRC文件
    mrc_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.mrc'):
                mrc_files.append(os.path.join(root, file))
    
    if not mrc_files:
        logger.warning(f"在 {input_dir} 中没有找到MRC文件")
        return [], []
    
    # 处理结果
    successful_files = []
    failed_files = []
    
    # 处理每个MRC文件
    for mrc_file in mrc_files:
        try:
            # 保留原始目录结构
            rel_path = os.path.relpath(os.path.dirname(mrc_file), input_dir)
            if rel_path == '.':  # 如果是根目录
                target_dir = output_dir
            else:
                target_dir = os.path.join(output_dir, rel_path)
            
            # 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 创建输出文件路径
            filename = os.path.basename(mrc_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_file = os.path.join(target_dir, f"{name_without_ext}.tif")
            
            # 转换文件
            mrc_to_tif(mrc_file, output_file)
            successful_files.append(mrc_file)
            
        except Exception as e:
            logger.error(f"处理文件 {mrc_file} 失败: {str(e)}")
            failed_files.append((mrc_file, str(e)))
    
    return successful_files, failed_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将MRC文件转换为TIF文件")
    parser.add_argument("input", help="输入MRC文件或目录")
    parser.add_argument("output", help="输出TIF文件或目录")
    parser.add_argument("--batch", action="store_true", help="批量处理目录")
    
    args = parser.parse_args()
    
    try:
        if args.batch or os.path.isdir(args.input):
            successful, failed = batch_convert_mrc_to_tif(args.input, args.output)
            print(f"批量转换完成，成功: {len(successful)}，失败: {len(failed)}")
            if failed:
                print("\n失败的文件:")
                for file, error in failed:
                    print(f"  - {file}: {error}")
        else:
            mrc_to_tif(args.input, args.output)
            print(f"成功将 {args.input} 转换为 {args.output}")
    except Exception as e:
        print(f"错误: {str(e)}") 