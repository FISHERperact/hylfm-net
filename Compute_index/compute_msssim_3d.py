#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import argparse
from pathlib import Path
import mrcfile
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from skimage.transform import resize

# 设置控制台和文件系统编码
if sys.platform.startswith('win'):
    # 设置控制台编码为UTF-8
    import subprocess
    subprocess.run(["chcp", "65001"], shell=True, check=False)

def read_3d_data(file_path):
    """
    读取3D数据，支持多种格式：mrc, tif, npy
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        ndarray: 3D数据数组
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.mrc':
        # 读取MRC文件
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data.copy()
    elif extension in ['.tif', '.tiff']:
        # 读取TIFF文件
        data = tifffile.imread(file_path)
    elif extension == '.npy':
        # 读取NumPy数组文件
        data = np.load(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {extension}")
    
    # 如果是2D图像，扩展为3D体
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    
    return data

def normalize_data(data):
    """
    归一化数据到[0,1]范围
    
    Args:
        data (ndarray): 输入数据
        
    Returns:
        ndarray: 归一化后的数据
    """
    data_min = data.min()
    data_max = data.max()
    
    if data_max > data_min:
        return (data - data_min) / (data_max - data_min)
    return data

def gaussian_filter(kernel_size=11, sigma=1.5, channels=1):
    """
    创建高斯滤波器
    
    Args:
        kernel_size (int): 滤波器大小
        sigma (float): 高斯方差
        channels (int): 通道数
        
    Returns:
        Tensor: 高斯滤波器
    """
    # 创建2D高斯核
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # 计算高斯核
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # 扩展为3D滤波器
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
    return gaussian_kernel

def compute_ssim_3d(img1, img2, window_size=11, sigma=1.5):
    """
    计算3D数据的SSIM
    
    Args:
        img1, img2 (Tensor): 输入的3D数据
        window_size (int): 窗口大小
        sigma (float): 高斯方差
        
    Returns:
        Tensor: SSIM值
    """
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
    
    # 确保数据是5D: [batch, channels, depth, height, width]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 4:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # 移动到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # 创建高斯窗口
    window = gaussian_filter(window_size, sigma)
    window = window.to(device)
    
    # 计算均值
    mu1 = torch.nn.functional.conv2d(
        img1.view(-1, 1, img1.size(-2), img1.size(-1)), window, 
        padding=window_size//2, groups=1
    )
    mu2 = torch.nn.functional.conv2d(
        img2.view(-1, 1, img2.size(-2), img2.size(-1)), window, 
        padding=window_size//2, groups=1
    )
    
    mu1 = mu1.view(img1.size(0), img1.size(1), img1.size(2), img1.size(3), img1.size(4))
    mu2 = mu2.view(img2.size(0), img2.size(1), img2.size(2), img2.size(3), img2.size(4))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = torch.nn.functional.conv2d(
        (img1 * img1).view(-1, 1, img1.size(-2), img1.size(-1)), window, 
        padding=window_size//2, groups=1
    ) - mu1_sq.view(-1, 1, mu1_sq.size(-2), mu1_sq.size(-1))
    
    sigma2_sq = torch.nn.functional.conv2d(
        (img2 * img2).view(-1, 1, img2.size(-2), img2.size(-1)), window, 
        padding=window_size//2, groups=1
    ) - mu2_sq.view(-1, 1, mu2_sq.size(-2), mu2_sq.size(-1))
    
    sigma12 = torch.nn.functional.conv2d(
        (img1 * img2).view(-1, 1, img1.size(-2), img1.size(-1)), window, 
        padding=window_size//2, groups=1
    ) - mu1_mu2.view(-1, 1, mu1_mu2.size(-2), mu1_mu2.size(-1))
    
    sigma1_sq = sigma1_sq.view(img1.size(0), img1.size(1), img1.size(2), img1.size(3), img1.size(4))
    sigma2_sq = sigma2_sq.view(img2.size(0), img2.size(1), img2.size(2), img2.size(3), img2.size(4))
    sigma12 = sigma12.view(img1.size(0), img1.size(1), img1.size(2), img1.size(3), img1.size(4))
    
    # SSIM公式常数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 对每个深度层取平均
    return ssim_map.mean(dim=(3, 4))  # 返回每个深度层的SSIM

def compute_msssim_3d(img1, img2, weights=None, levels=5):
    """
    计算3D数据的MS-SSIM
    
    Args:
        img1, img2 (ndarray): 输入的3D数据
        weights (list): 多尺度权重
        levels (int): 多尺度层级数
        
    Returns:
        float: MS-SSIM值
    """
    # 归一化数据到[0,1]
    img1 = normalize_data(img1)
    img2 = normalize_data(img2)
    
    # 转换为torch.Tensor
    img1 = torch.from_numpy(img1).float()
    img2 = torch.from_numpy(img2).float()
    
    # 默认权重
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    else:
        weights = torch.tensor(weights)
    
    # 确保层级不超过图像大小限制
    min_size = min(min(img1.shape[-2:]), 2 ** (levels - 1))
    levels = min(levels, int(np.log2(min_size)))
    
    # 计算MS-SSIM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = weights[:levels].to(device)
    
    msssim_values = []
    
    # 遍历每个深度层
    for z in range(img1.shape[0]):
        msssim_per_slice = []
        img1_slice = img1[z:z+1]
        img2_slice = img2[z:z+1]
        
        for level in range(levels):
            ssim_slice = compute_ssim_3d(img1_slice, img2_slice)[0, 0, 0]
            msssim_per_slice.append(ssim_slice.item())
            
            # 降采样，除最后一级外
            if level < levels - 1:
                img1_slice = torch.nn.functional.avg_pool2d(img1_slice, kernel_size=2)
                img2_slice = torch.nn.functional.avg_pool2d(img2_slice, kernel_size=2)
        
        # 计算加权乘积
        msssim_slice = np.prod(np.power(msssim_per_slice, weights.cpu().numpy()))
        msssim_values.append(msssim_slice)
    
    # 所有深度层的平均MS-SSIM
    return np.mean(msssim_values)

def compare_volumes(file1, file2):
    """
    比较两个3D体数据的MS-SSIM
    
    Args:
        file1, file2 (str): 两个3D数据文件路径
        
    Returns:
        float: MS-SSIM值
    """
    print(f"读取第一个文件: {file1}")
    data1 = read_3d_data(file1)
    
    print(f"读取第二个文件: {file2}")
    data2 = read_3d_data(file2)
    
    # 处理4D数据，如果最后一维是1，则去掉
    if data1.ndim == 4 and data1.shape[-1] == 1:
        data1 = data1[..., 0]
    if data2.ndim == 4 and data2.shape[-1] == 1:
        data2 = data2[..., 0]
    
    # 检查形状是否匹配
    if data1.shape != data2.shape:
        print(f"警告: 文件形状不匹配 - {data1.shape} vs {data2.shape}")
        
        # 确保深度维度相同
        if data1.shape[0] != data2.shape[0]:
            min_depth = min(data1.shape[0], data2.shape[0])
            data1 = data1[:min_depth]
            data2 = data2[:min_depth]
        
        # 找到较小的空间尺寸
        target_height = min(data1.shape[1], data2.shape[1])
        target_width = min(data1.shape[2], data2.shape[2])
        
        # 对较大的图像进行缩放
        if data1.shape[1] > target_height or data1.shape[2] > target_width:
            print(f"缩放第一个数据到: ({data1.shape[0]}, {target_height}, {target_width})")
            data1_resized = np.zeros((data1.shape[0], target_height, target_width))
            for z in range(data1.shape[0]):
                data1_resized[z] = resize(data1[z], (target_height, target_width), 
                                        preserve_range=True, anti_aliasing=True)
            data1 = data1_resized
            
        if data2.shape[1] > target_height or data2.shape[2] > target_width:
            print(f"缩放第二个数据到: ({data2.shape[0]}, {target_height}, {target_width})")
            data2_resized = np.zeros((data2.shape[0], target_height, target_width))
            for z in range(data2.shape[0]):
                data2_resized[z] = resize(data2[z], (target_height, target_width),
                                        preserve_range=True, anti_aliasing=True)
            data2 = data2_resized
        
        print(f"调整后的形状: {data1.shape}")
    
    print(f"计算MS-SSIM (形状: {data1.shape})...")
    msssim_value = compute_msssim_3d(data1, data2)
    
    return msssim_value

def create_comparison_report(file1, file2, output_dir=None):
    """
    创建两个3D体数据的比较报告
    
    Args:
        file1, file2 (str): 两个3D数据文件路径
        output_dir (str): 输出目录
        
    Returns:
        str: 报告文件路径
    """
    # 计算MS-SSIM
    msssim = compare_volumes(file1, file2)
    print(f"MS-SSIM: {msssim:.6f}")
    
    # 读取数据用于可视化
    data1 = read_3d_data(file1)
    data2 = read_3d_data(file2)
    
    # 处理4D数据，如果最后一维是1，则去掉
    if data1.ndim == 4 and data1.shape[-1] == 1:
        data1 = data1[..., 0]
    if data2.ndim == 4 and data2.shape[-1] == 1:
        data2 = data2[..., 0]
    
    # 确保形状匹配
    if data1.shape != data2.shape:
        # 确保深度维度相同
        if data1.shape[0] != data2.shape[0]:
            min_depth = min(data1.shape[0], data2.shape[0])
            data1 = data1[:min_depth]
            data2 = data2[:min_depth]
        
        # 找到较小的空间尺寸
        target_height = min(data1.shape[1], data2.shape[1])
        target_width = min(data1.shape[2], data2.shape[2])
        
        # 对较大的图像进行缩放
        if data1.shape[1] > target_height or data1.shape[2] > target_width:
            data1_resized = np.zeros((data1.shape[0], target_height, target_width))
            for z in range(data1.shape[0]):
                data1_resized[z] = resize(data1[z], (target_height, target_width), 
                                        preserve_range=True, anti_aliasing=True)
            data1 = data1_resized
            
        if data2.shape[1] > target_height or data2.shape[2] > target_width:
            data2_resized = np.zeros((data2.shape[0], target_height, target_width))
            for z in range(data2.shape[0]):
                data2_resized[z] = resize(data2[z], (target_height, target_width),
                                        preserve_range=True, anti_aliasing=True)
            data2 = data2_resized
    
    # 计算差异图
    diff = np.abs(normalize_data(data1) - normalize_data(data2))
    
    # 创建输出目录
    if output_dir is None:
        output_dir = 'ms_ssim_比较结果'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建报告文件名
    file1_name = os.path.basename(file1).split('.')[0]
    file2_name = os.path.basename(file2).split('.')[0]
    report_file = os.path.join(output_dir, f"比较报告_{file1_name}_vs_{file2_name}.txt")
    
    # 写入报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"3D体数据MS-SSIM比较报告\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"文件1: {file1}\n")
        f.write(f"文件2: {file2}\n\n")
        f.write(f"原始形状:\n")
        f.write(f"文件1: {data1.shape}\n")
        f.write(f"文件2: {data2.shape}\n")
        f.write(f"MS-SSIM: {msssim:.6f}\n")
        f.write(f"{'='*50}\n")
        
        # 添加评价标准
        f.write("\nMS-SSIM评价标准:\n")
        f.write(" - 1.0: 完全相同\n")
        f.write(" - 0.9-0.99: 极其相似，几乎无法区分\n")
        f.write(" - 0.8-0.9: 非常相似，仅有微小差异\n")
        f.write(" - 0.7-0.8: 相似，但有明显差异\n")
        f.write(" - 0.6-0.7: 中等相似度\n")
        f.write(" - <0.6: 较大差异\n")
    
    # 生成可视化比较图
    center_slice = data1.shape[0] // 2
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.imshow(normalize_data(data1[center_slice]), cmap='gray')
    plt.title(f"文件1 - 中心切片")
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(normalize_data(data2[center_slice]), cmap='gray')
    plt.title(f"文件2 - 中心切片")
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(diff[center_slice], cmap='hot')
    plt.title(f"差异图 (MS-SSIM: {msssim:.4f})")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"比较可视化_{file1_name}_vs_{file2_name}.png"), dpi=150)
    plt.close()
    
    print(f"比较报告已保存到: {report_file}")
    return report_file

def main():
    # 设置默认路径
    file1 = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\光场重建结果\ER\00000_c002d_2048_3d2(7)_MS-SSIM.tif"
    file2 = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\ER\ER(OK)\WF_ER\sample (181).tif"
    output_dir = 'ms_ssim_比较结果'
    
    # 检查文件是否存在
    if not os.path.exists(file1):
        print(f"错误: 文件不存在 - {file1}")
        return
    
    if not os.path.exists(file2):
        print(f"错误: 文件不存在 - {file2}")
        return
    
    # 计算MS-SSIM并生成报告
    create_comparison_report(file1, file2, output_dir)

if __name__ == "__main__":
    main() 