#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm
import gc  # 垃圾回收模块，用于释放内存

def split_tif_volume(input_file, output_dir, layers_per_split=21):
    """
    将3D TIF文件切分为多个相同宽高但深度为固定层数的TIF文件
    
    参数:
    - input_file: 输入TIF文件路径
    - output_dir: 输出目录路径
    - layers_per_split: 每个输出文件的层数
    
    返回:
    - 保存的输出文件路径列表
    """
    try:
        # 读取输入TIF文件
        print(f"读取文件: {os.path.basename(input_file)}")
        input_data = tifffile.imread(input_file)
        
        # 检查输入数据是否为3D
        if len(input_data.shape) != 3:
            print(f"警告: {os.path.basename(input_file)} 不是3D数据，跳过处理")
            return []
        
        # 获取输入数据尺寸
        depth, height, width = input_data.shape
        print(f"输入数据尺寸: {depth}×{height}×{width}")
        
        # 如果深度小于等于layers_per_split，无需切分
        if depth <= layers_per_split:
            print(f"警告: {os.path.basename(input_file)} 深度 ({depth}) 不足以切分，跳过处理")
            return []
        
        # 计算可以切分的份数
        num_splits = 3  # 固定切分为3份
        
        # 确定每个切片的起始和结束索引
        # 选择前、中、后三个区域的策略
        split_indices = []
        
        # 前21层
        split_indices.append((0, layers_per_split))
        
        # 中间21层
        middle_start = (depth - layers_per_split) // 2
        split_indices.append((middle_start, middle_start + layers_per_split))
        
        # 后21层
        end_start = depth - layers_per_split
        split_indices.append((end_start, depth))
        
        # 创建输出目录
        output_path = Path(output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True)
            print(f"创建输出目录: {output_dir}")
        
        output_files = []
        
        # 切分并保存每个部分
        for i, (start_idx, end_idx) in enumerate(split_indices):
            # 提取对应层的数据
            split_data = input_data[start_idx:end_idx]
            
            # 构建输出文件名
            input_filename = Path(input_file).stem
            region_name = "top" if i == 0 else "middle" if i == 1 else "bottom"
            output_file = output_path / f"{input_filename}_{region_name}_layers_{start_idx+1}-{end_idx}.tif"
            
            # 保存为TIF文件
            print(f"保存切片 {i+1}/{num_splits}: {output_file.name}")
            tifffile.imwrite(output_file, split_data)
            output_files.append(str(output_file))
            
            # 释放内存
            del split_data
            gc.collect()
        
        # 释放输入数据内存
        del input_data
        gc.collect()
        
        print(f"成功将 {os.path.basename(input_file)} 切分为 {len(output_files)} 个文件")
        return output_files
    
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return []

def process_directory(input_dir, output_dir, recursive=True):
    """
    处理指定目录下的所有TIF文件，将每个文件切分为三个部分
    
    参数:
    - input_dir: 输入目录路径
    - output_dir: 输出目录路径
    - recursive: 是否递归处理子目录
    """
    # 转换为Path对象
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 如果输出目录不存在，创建它
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有TIF文件
    pattern = "**/*.tif" if recursive else "*.tif"
    tif_files = list(input_path.glob(pattern))
    
    if not tif_files:
        print(f"在 {input_dir} 中没有找到TIF文件")
        return
    
    print(f"在 {input_dir} 中找到 {len(tif_files)} 个TIF文件")
    
    # 处理统计
    successful_files = 0
    processed_files = 0
    
    # 处理每个TIF文件
    for tif_file in tqdm(tif_files, desc="切分TIF文件"):
        # 构建输出子目录路径（保持与输入文件相同的相对路径结构）
        rel_path = tif_file.relative_to(input_path).parent
        file_output_dir = output_path / rel_path
        
        # 切分文件
        output_files = split_tif_volume(tif_file, file_output_dir)
        
        processed_files += 1
        if output_files:
            successful_files += 1
        
        # 手动触发垃圾回收
        gc.collect()
    
    # 打印处理统计
    print("\n处理完成!")
    print(f"成功处理: {successful_files}/{processed_files} 个文件")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    # 设置输入和输出目录
    input_directory = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\silk\Silk(RAW)"
    output_directory = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\silk\Silk(RAW)_Split"
    
    # 处理目录
    process_directory(input_directory, output_directory)