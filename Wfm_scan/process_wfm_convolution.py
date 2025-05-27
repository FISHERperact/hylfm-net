#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import datetime
import numpy as np
import mrcfile
from pathlib import Path
from scipy.signal import convolve2d
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def read_mrc_file(mrc_path):
    """
    读取MRC文件并返回数据数组
    
    参数:
    - mrc_path: MRC文件路径
    
    返回:
    - data: numpy数组，形状为(z, y, x)
    """
    try:
        with mrcfile.open(mrc_path, permissive=True, mode='r') as mrc:
            # 转换为float64以确保卷积运算的精度
            data = mrc.data.astype(np.float64)
        return data
    except Exception as e:
        print(f"读取MRC文件 {mrc_path} 时出错: {str(e)}")
        return None

def read_psf(psf_path):
    """
    读取PSF文件并返回归一化的PSF数组
    
    参数:
    - psf_path: PSF TIF文件路径
    
    返回:
    - psf: 归一化的PSF numpy数组
    """
    try:
        psf = imread(psf_path)
        psf = psf.astype(np.float64)
        # 归一化PSF，使其最大值为1
        psf = psf / np.max(psf)
        return psf
    except Exception as e:
        print(f"读取PSF文件 {psf_path} 时出错: {str(e)}")
        return None

def apply_wfm_convolution(image_data, psf):
    """
    对每一层应用宽场卷积
    
    参数:
    - image_data: 3D图像数据，形状为(z, y, x)
    - psf: 点扩散函数
    
    返回:
    - wfm_data: 卷积后的宽场图像数据
    """
    # 获取图像数据的维度
    z_dim, y_dim, x_dim = image_data.shape
    
    # 创建输出数组
    wfm_data = np.zeros_like(image_data)
    
    # 对每一层应用卷积
    for z in range(z_dim):
        # 获取当前层
        current_slice = image_data[z, :, :]
        
        # 应用卷积
        convolved_slice = convolve2d(current_slice, psf, mode='same')
        
        # 归一化处理后的数据
        if np.max(convolved_slice) > 0:  # 避免除以0
            convolved_slice = convolved_slice / np.max(convolved_slice)
        
        # 保存到输出数组
        wfm_data[z, :, :] = convolved_slice
    
    return wfm_data

def save_data_as_tiff(data, output_path, normalize=True):
    """
    将数据保存为TIFF文件
    
    参数:
    - data: 要保存的数据
    - output_path: 输出文件路径
    - normalize: 是否在保存前归一化数据
    """
    try:
        # 如果需要归一化
        if normalize and np.max(data) > 0:
            data = data / np.max(data)
        
        # 转换为16位整数以减小文件大小
        data = (data * 65535).astype(np.uint16)
        
        # 保存为TIFF
        imwrite(output_path, data)
        return True
    except Exception as e:
        print(f"保存TIFF文件 {output_path} 时出错: {str(e)}")
        return False

def visualize_slices(data, output_dir, base_filename, max_slices=21):
    """
    可视化数据的切片并保存为图像
    
    参数:
    - data: 3D数据
    - output_dir: 输出目录
    - base_filename: 基础文件名
    - max_slices: 最大切片数量
    """
    z_dim = min(data.shape[0], max_slices)
    
    # 计算网格布局
    grid_size = int(np.ceil(np.sqrt(z_dim)))
    
    plt.figure(figsize=(15, 15))
    for i in range(z_dim):
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(data[i], cmap='viridis')
        plt.title(f'Slice {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_slices.png"), dpi=150)
    plt.close()

def process_mrc_files(source_dir, psf_path, output_dir, visualize=False):
    """
    处理目录中的所有MRC文件，应用宽场卷积，并保存结果
    
    参数:
    - source_dir: 包含MRC文件的目录
    - psf_path: PSF文件路径
    - output_dir: 输出目录
    - visualize: 是否生成可视化图像
    """
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    # 读取PSF
    psf = read_psf(psf_path)
    if psf is None:
        print(f"无法继续，因为无法读取PSF文件 {psf_path}")
        return
    
    # 使用pathlib递归查找所有.mrc文件
    source_path = Path(source_dir)
    print(f"搜索 {source_dir} 下的所有.mrc文件...")
    mrc_files = list(source_path.glob("**/*.mrc"))  # 递归搜索所有子目录
    
    if not mrc_files:
        print(f"警告: 在 {source_dir} 下没有找到.mrc文件")
        return
    
    # 按文件名排序，保证处理顺序
    mrc_files.sort()
    
    print(f"找到 {len(mrc_files)} 个.mrc文件")
    
    # 处理每个MRC文件
    processed_count = 0
    
    for mrc_file in mrc_files:
        # 获取文件名（不含扩展名）
        base_name = mrc_file.stem
        
        # 获取子目录结构（相对于source_dir）
        rel_path = mrc_file.relative_to(source_path).parent
        # 构建对应的输出子目录
        output_subdir = os.path.join(output_dir, str(rel_path))
        ensure_dir_exists(output_subdir)
        
        print(f"处理文件: {mrc_file}")
        
        # 读取MRC数据
        mrc_data = read_mrc_file(str(mrc_file))
        if mrc_data is None:
            continue
        
        # 应用宽场卷积
        wfm_data = apply_wfm_convolution(mrc_data, psf)
        
        # 保存为TIFF文件
        output_file = os.path.join(output_subdir, f"{base_name}_wfm.tif")
        success = save_data_as_tiff(wfm_data, output_file)
        
        if success:
            processed_count += 1
            print(f"成功处理并保存到: {output_file}")
            
            # 可视化（如果需要）
            if visualize:
                visualize_slices(wfm_data, output_subdir, f"{base_name}_wfm", max_slices=21)
    
    print(f"\n处理完成! 成功处理 {processed_count}/{len(mrc_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='对MRC文件应用宽场卷积')
    
    # 参数设置
    parser.add_argument('--source-dir', '-s', type=str, 
                       default=r'D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\CCPs(raw)\CCPs\Cell_002',
                       help='包含MRC文件的源目录')
    parser.add_argument('--psf-path', '-p', type=str, 
                       default='D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\宽场psf\psf_wfm.tif',
                       help='PSF文件路径')
    parser.add_argument('--output-dir', '-o', type=str, 
                       default=r'D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\CCPs(raw)\CCPs\Cell_002\WFM',
                       help='输出目录路径')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='是否生成可视化图像')
    
    args = parser.parse_args()
    
    # 处理MRC文件
    process_mrc_files(args.source_dir, args.psf_path, args.output_dir, args.visualize)

if __name__ == "__main__":
    main() 