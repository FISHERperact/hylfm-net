#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import mrcfile
import argparse
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, morphology, feature, measure
import cv2
import time

def read_mrc(file_path):
    """读取MRC文件并返回数据"""
    try:
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data.copy()
            # 如果是3D数据，取第一层作为2D数据
            if len(data.shape) == 3:
                data = data[0]
            return data
    except Exception as e:
        print(f"读取MRC文件时出错: {e}")
        return None

def read_image(file_path):
    """读取图像文件（支持多种格式）"""
    if file_path.lower().endswith('.mrc'):
        return read_mrc(file_path)
    else:
        try:
            # 使用OpenCV读取常见图像格式
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise Exception("无法读取图像")
            return img
        except Exception as e:
            print(f"读取图像文件时出错: {e}")
            return None

def extract_lines(image, method='ridge', threshold=0.05, line_width=3):
    """
    从图像中提取线条
    
    参数:
    - image: 输入图像
    - method: 线条提取方法 ('canny', 'threshold', 'ridge')
    - threshold: 阈值参数
    - line_width: 线条宽度
    
    返回:
    - lines_image: 提取的线条图像
    """
    # 确保图像是浮点型，并归一化到[0,1]
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
    
    if method == 'canny':
        # Canny边缘检测
        edges = feature.canny(image, sigma=1.0, low_threshold=threshold, high_threshold=threshold*2)
        # 膨胀边缘以增加线条宽度
        if line_width > 1:
            edges = morphology.dilation(edges, morphology.disk(line_width//2))
        return edges.astype(np.float32)
    
    elif method == 'threshold':
        # 简单阈值处理
        binary = image > threshold
        # 可选择性地进行形态学操作以增强线条
        if line_width > 1:
            binary = morphology.dilation(binary, morphology.disk(line_width//2))
        return binary.astype(np.float32)
    
    elif method == 'ridge':
        # 使用Hessian矩阵的特征值检测脊线
        ridges = filters.frangi(image, sigmas=range(1, 5), black_ridges=False)
        # 阈值处理
        ridges_binary = ridges > threshold * np.max(ridges)
        # 增加线条宽度
        if line_width > 1:
            ridges_binary = morphology.dilation(ridges_binary, morphology.disk(line_width//2))
        return ridges_binary.astype(np.float32)
    
    else:
        print(f"未知的线条提取方法: {method}")
        return image

def create_3d_from_lines(lines_image, num_layers=21, z_distribution='linear', 
                         tilt_factor=0.7, z_thickness=3, intensity_factor=1.5, z_blur=1.0, 
                         random_factor=1.0, tilt_preserve=0.7, z_stretch=1.0,
                         wave_frequency=2.0, wave_amplitude=0.8, use_fixed_seed=False):
    """
    将2D线条图像转换为3D体积
    
    参数:
    - lines_image: 提取的线条图像
    - num_layers: 3D体积的层数
    - z_distribution: Z方向分布方式 ('linear', 'gaussian', 'distance_transform', 'random', 'wave')
    - tilt_factor: 控制Z方向变化的因子
    - z_thickness: Z方向上每条线的厚度
    - intensity_factor: 线条亮度因子
    - z_blur: Z方向模糊程度 (值越大越模糊)
    - random_factor: 随机程度 (值越大越随机，仅用于random分布方式)
    - tilt_preserve: 倾斜保持系数 (0-1之间，值越大保留的倾斜趋势越明显，仅用于random分布方式)
    - z_stretch: Z轴拉伸系数 (值越大Z轴拉伸效果越明显)
    - wave_frequency: 波浪频率 (值越大，波动越频繁)
    - wave_amplitude: 波浪幅度 (0-1之间，值越大波动越剧烈)
    - use_fixed_seed: 是否使用固定随机种子以确保可重复性
    
    返回:
    - volume_3d: 3D体积数据
    """
    start_time = time.time()
    
    height, width = lines_image.shape
    
    # 应用Z轴拉伸，增加层数
    stretched_layers = int(num_layers)
    volume_3d = np.zeros((stretched_layers, height, width), dtype=np.float32)
    
    # 创建中心层索引
    center_layer = stretched_layers // 2
    
    # 计算高斯衰减的方差，z_blur控制模糊程度
    # 增加z_blur以获得更模糊的效果
    variance = (z_thickness/2)**2 * z_blur
    
    # 获取所有非零点的坐标，用于进度计算和处理
    y_coords, x_coords = np.where(lines_image > 0)
    total_points = len(y_coords)
    
    if total_points == 0:
        print("警告: 没有检测到线条点，返回空体积")
        return volume_3d
    
    print(f"开始处理 {total_points} 个非零点...")
    progress_step = max(1, total_points // 10)  # 每10%显示一次进度
    last_progress = 0
    
    if z_distribution == 'linear':
        # 基于位置的线性Z分布
        y_norm = np.linspace(0, 1, height)
        x_norm = np.linspace(0, 1, width)
        Y, X = np.meshgrid(y_norm, x_norm, indexing='ij')
        
        # 计算Z值 (基于X和Y的位置)
        # 应用Z轴拉伸调整Z分布
        Z = (X + Y) * tilt_factor * num_layers * z_stretch
        Z = Z.astype(int) % stretched_layers
        
        # 对于每个非零点，在Z方向上创建厚度
        for point_idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            z_center = Z[y, x]
            
            # 在更广范围内应用高斯衰减
            z_range = int(z_thickness * z_blur * z_stretch)
            for z_offset in range(-z_range, z_range + 1):
                z = (z_center + z_offset) % stretched_layers  # 确保z在有效范围内
                # 使用高斯衰减使中心更亮
                intensity = np.exp(-(z_offset**2) / (2 * variance * z_stretch))
                volume_3d[z, y, x] = lines_image[y, x] * intensity_factor * intensity
            
            # 显示进度
            if point_idx % progress_step == 0:
                progress = (point_idx / total_points) * 100
                elapsed = time.time() - start_time
                if progress > last_progress + 9:  # 每增加10%显示一次
                    print(f"处理进度: {progress:.1f}% ({point_idx}/{total_points}) 耗时: {elapsed:.1f}秒")
                    last_progress = progress
    
    elif z_distribution == 'gaussian':
        # 在Z方向上使用高斯分布
        for point_idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            # 基于位置计算Z中心 (应用Z轴拉伸)
            z_center = int((y / height + x / width) * tilt_factor * num_layers * z_stretch) % stretched_layers
            
            # 在Z方向上创建高斯分布
            for z in range(stretched_layers):
                # 计算到中心的距离
                z_dist = min(abs(z - z_center), abs(z - z_center - stretched_layers), abs(z - z_center + stretched_layers))
                # 高斯衰减，使用增强的模糊度
                intensity = np.exp(-(z_dist**2) / (2 * variance * z_stretch))
                volume_3d[z, y, x] = lines_image[y, x] * intensity_factor * intensity
            
            # 显示进度
            if point_idx % progress_step == 0:
                progress = (point_idx / total_points) * 100
                elapsed = time.time() - start_time
                if progress > last_progress + 9:
                    print(f"处理进度: {progress:.1f}% ({point_idx}/{total_points}) 耗时: {elapsed:.1f}秒")
                    last_progress = progress
    
    elif z_distribution == 'distance_transform':
        # 使用距离变换来确定Z值
        # 计算到最近非线条点的距离
        dist_transform = ndimage.distance_transform_edt(lines_image > 0)
        # 归一化距离
        dist_norm = dist_transform / (dist_transform.max() + 1e-10)
        
        for point_idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            # 基于距离变换和位置计算Z中心 (应用Z轴拉伸)
            pos_factor = (y / height + x / width) / 2
            dist_factor = dist_norm[y, x]
            z_center = int((pos_factor + dist_factor * 0.5) * tilt_factor * num_layers * z_stretch) % stretched_layers
            
            # 在更广范围内应用高斯衰减
            z_range = int(z_thickness * z_blur * z_stretch)
            for z_offset in range(-z_range, z_range + 1):
                z = (z_center + z_offset) % stretched_layers  # 确保z在有效范围内
                # 使用高斯衰减使中心更亮
                intensity = np.exp(-(z_offset**2) / (2 * variance * z_stretch))
                volume_3d[z, y, x] = lines_image[y, x] * intensity_factor * intensity
            
            # 显示进度
            if point_idx % progress_step == 0:
                progress = (point_idx / total_points) * 100
                elapsed = time.time() - start_time
                if progress > last_progress + 9:
                    print(f"处理进度: {progress:.1f}% ({point_idx}/{total_points}) 耗时: {elapsed:.1f}秒")
                    last_progress = progress
    
    elif z_distribution == 'random':
        # 使用半随机分布方式，为每个线条点生成带有倾斜趋势的随机Z值
        print(f"使用随机Z分布方式 (随机因子: {random_factor}, 倾斜保持: {tilt_preserve}, Z轴拉伸: {z_stretch})...")
        
        # 提前计算所有可能的点的Z值
        # 1. 创建基础Z值 (基于位置的线性分布)
        y_norm = np.linspace(0, 1, height)
        x_norm = np.linspace(0, 1, width)
        Y, X = np.meshgrid(y_norm, x_norm, indexing='ij')
        base_Z = ((X + Y) * tilt_factor * num_layers * z_stretch).astype(int) % stretched_layers
        
        # 2. 生成随机偏移
        # 使用随机因子调整随机程度，random_factor越大，随机性越强
        if use_fixed_seed:
            np.random.seed(42)  # 使用固定种子以确保可重复性
        random_offset = (np.random.rand(height, width) * 2 - 1) * random_factor * num_layers * z_stretch / 4
        
        # 3. 将基础Z值和随机偏移结合，使用tilt_preserve参数控制保留多少基础倾斜
        Z = (base_Z * tilt_preserve + random_offset.astype(int) * (1 - tilt_preserve)) % stretched_layers
        
        # 4. 使用高斯模糊平滑随机分布，使相邻点的Z值更加连续
        # 当随机因子较小或倾斜保持较高时，平滑程度较低
        smoothing_factor = 2.0 / (random_factor + tilt_preserve)
        Z = ndimage.gaussian_filter(Z, sigma=smoothing_factor)
        Z = Z.astype(int) % stretched_layers
        
        # 对于每个非零点，在Z方向上创建厚度
        for point_idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            z_center = Z[y, x]
            
            # 在更广范围内应用高斯衰减
            z_range = int(z_thickness * z_blur * z_stretch)
            for z_offset in range(-z_range, z_range + 1):
                z = (z_center + z_offset) % stretched_layers  # 确保z在有效范围内
                # 使用高斯衰减使中心更亮
                intensity = np.exp(-(z_offset**2) / (2 * variance * z_stretch))
                volume_3d[z, y, x] = lines_image[y, x] * intensity_factor * intensity
            
            # 显示进度
            if point_idx % progress_step == 0:
                progress = (point_idx / total_points) * 100
                elapsed = time.time() - start_time
                if progress > last_progress + 9:
                    print(f"处理进度: {progress:.1f}% ({point_idx}/{total_points}) 耗时: {elapsed:.1f}秒")
                    last_progress = progress
    
    elif z_distribution == 'wave':
        # 使用自然随机分布方式，为每个线条点生成随机但连续的Z值
        print(f"使用随机波动Z分布方式 (频率: {wave_frequency}, 幅度: {wave_amplitude}, Z轴拉伸: {z_stretch})...")
        
        # 创建基础位置坐标
        y_norm = np.linspace(0, 1, height)
        x_norm = np.linspace(0, 1, width)
        Y, X = np.meshgrid(y_norm, x_norm, indexing='ij')
        
        # 创建基础Z值
        # 使用更加自然的随机分布而不是正弦波
        # 使用多个不同频率和振幅的噪声叠加，创建更自然的随机效果
        # 移除固定种子，使每次运行生成不同结果
        if use_fixed_seed:
            np.random.seed(42)  # 使用固定种子以确保可重复性
        
        # 创建多层噪声
        noise_base = np.zeros((height, width))
        
        # 创建大尺度噪声（低频变化）
        scale1 = int(width * 0.2)  # 大尺度变化
        noise1 = np.random.rand(height // scale1 + 1, width // scale1 + 1)
        noise1 = ndimage.zoom(noise1, (height / noise1.shape[0], width / noise1.shape[1]), order=1)
        
        # 创建中尺度噪声（中频变化）
        scale2 = int(width * 0.1)  # 中尺度变化
        noise2 = np.random.rand(height // scale2 + 1, width // scale2 + 1)
        noise2 = ndimage.zoom(noise2, (height / noise2.shape[0], width / noise2.shape[1]), order=1)
        
        # 创建小尺度噪声（高频变化）
        scale3 = int(width * 0.05)  # 小尺度变化
        noise3 = np.random.rand(height // scale3 + 1, width // scale3 + 1)
        noise3 = ndimage.zoom(noise3, (height / noise3.shape[0], width / noise3.shape[1]), order=1)
        
        # 按权重混合多层噪声，保持频率参数的影响力
        # 频率越高，越强调高频细节
        freq_factor = min(max(wave_frequency, 0.5), 3.0)  # 约束在合理范围
        
        weight1 = 0.6 / freq_factor  # 大尺度权重随频率增加而减小
        weight2 = 0.3 * freq_factor  # 中尺度权重随频率增加而增加
        weight3 = 0.1 * freq_factor  # 小尺度权重随频率增加而增加
        
        # 归一化权重
        total_weight = weight1 + weight2 + weight3
        weight1 /= total_weight
        weight2 /= total_weight
        weight3 /= total_weight
        
        # 混合噪声
        noise_base = weight1 * noise1 + weight2 * noise2 + weight3 * noise3
        
        # 调整波动幅度
        # 将噪声值从[0,1]归一化到[-1,1]，然后乘以幅度参数控制强度
        noise_base = (noise_base * 2 - 1) * wave_amplitude
        
        # 添加基于Y坐标的倾斜因素，但减小其影响，防止分离
        tilt_effect = Y * tilt_factor * 0.7 * stretched_layers
        
        # 计算Z值，使用中心的一段空间，避免取模导致的边缘折返
        # 使用stretched_layers的中间部分以避免在Z方向上出现断裂
        z_min = stretched_layers // 4  # 从1/4处开始
        z_range = stretched_layers // 2  # 使用1/2的范围
        
        # 归一化噪声到一定的z范围内，确保不会有大的分离
        Z = z_min + (noise_base + 0.5) * z_range
        Z = Z + tilt_effect  # 添加倾斜效果
        
        # 确保Z值在有效范围内，但避免使用取模操作
        # 而是将超出部分裁剪到边界
        Z = np.clip(Z, 0, stretched_layers - 1)
        
        # 应用高斯平滑使Z值变化更连续自然
        Z = ndimage.gaussian_filter(Z, sigma=2.0)
        Z = Z.astype(int)
        
        # 对于每个非零点，在Z方向上创建厚度
        for point_idx, (y, x) in enumerate(zip(y_coords, x_coords)):
            z_center = Z[y, x]
            
            # 在Z方向上创建厚度
            z_range_local = int(z_thickness * z_blur * z_stretch)
            for z_offset in range(-z_range_local, z_range_local + 1):
                z = np.clip(z_center + z_offset, 0, stretched_layers - 1)  # 确保z在有效范围内，不使用取模
                # 使用高斯衰减使中心更亮
                intensity = np.exp(-(z_offset**2) / (2 * variance * z_stretch))
                volume_3d[z, y, x] = lines_image[y, x] * intensity_factor * intensity
            
            # 显示进度
            if point_idx % progress_step == 0:
                progress = (point_idx / total_points) * 100
                elapsed = time.time() - start_time
                if progress > last_progress + 9:
                    print(f"处理进度: {progress:.1f}% ({point_idx}/{total_points}) 耗时: {elapsed:.1f}秒")
                    last_progress = progress
    
    # 可选: 在Z方向上应用高斯模糊以进一步增强模糊效果
    if z_blur > 1.5:  # 只有当模糊度较高时才应用额外的模糊
        print("应用Z方向高斯模糊...")
        sigma = (z_blur - 1.0) * 0.5  # 基于z_blur计算高斯模糊的sigma
        volume_3d = ndimage.gaussian_filter1d(volume_3d, sigma=sigma, axis=0)
    
    total_elapsed = time.time() - start_time
    print(f"3D体积生成完成，总耗时: {total_elapsed:.1f}秒")
    return volume_3d

def save_mrc(data, output_path):
    """将数据保存为MRC文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(np.float32))
        print(f"已保存3D数据到: {output_path}")
        return True
    except Exception as e:
        print(f"保存MRC文件时出错: {e}")
        return False

def save_preview(original_image, lines_image, volume_3d, output_prefix):
    """保存预览图像"""
    # 创建图像目录
    preview_dir = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(preview_dir, exist_ok=True)
    
    # 旋转图像180度并左右镜像翻转
    original_image = cv2.rotate(original_image, cv2.ROTATE_180)
    original_image = cv2.flip(original_image, 1)  # 1表示左右翻转
    
    lines_image = cv2.rotate(lines_image, cv2.ROTATE_180)
    lines_image = cv2.flip(lines_image, 1)  # 1表示左右翻转
    
    volume_3d = np.rot90(volume_3d, k=2, axes=(1, 2))  # 在XY平面上旋转180度
    volume_3d = np.flip(volume_3d, axis=2)  # 在X轴方向（左右）翻转
    
    # 1. 原始图像和提取的线条对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(lines_image, cmap='gray')
    plt.title('提取的线条')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_lines_extraction.png", dpi=150)
    plt.close()
    
    # 2. 3D体积的不同层预览
    num_layers = volume_3d.shape[0]
    preview_layers = min(5, num_layers)
    indices = np.linspace(0, num_layers-1, preview_layers).astype(int)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, preview_layers, i+1)
        plt.imshow(volume_3d[idx], cmap='gray')
        plt.title(f'层 {idx}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_3d_layers.png", dpi=150)
    plt.close()
    
    # 3. 3D体积的最大投影 - 改进Z轴显示
    max_proj_xy = np.max(volume_3d, axis=0)  # XY平面最大投影
    max_proj_xz = np.max(volume_3d, axis=1)  # XZ平面最大投影
    max_proj_yz = np.max(volume_3d, axis=2)  # YZ平面最大投影
    
    # 优化显示比例，拉伸Z轴方向
    # 计算合适的高宽比，使Z轴效果更明显
    z_layers = volume_3d.shape[0]
    height = volume_3d.shape[1]
    width = volume_3d.shape[2]
    
    # 计算Z轴比例，确保Z轴不会太扁
    aspect_ratio_xz = width / (z_layers * 2)  # 使Z轴看起来更高
    aspect_ratio_yz = height / (z_layers * 2)
    
    plt.figure(figsize=(15, 7))  # 增加图像高度
    
    plt.subplot(1, 3, 1)
    plt.imshow(max_proj_xy, cmap='gray')
    plt.title('XY平面最大投影')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(max_proj_xz, cmap='gray', aspect=aspect_ratio_xz)
    plt.title('XZ平面最大投影 (Z轴增强)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(max_proj_yz, cmap='gray', aspect=aspect_ratio_yz)
    plt.title('YZ平面最大投影 (Z轴增强)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_3d_projections.png", dpi=150)
    plt.close()
    
    # 4. 添加3D体积的垂直切片预览（新增）
    # 选择中心切片
    center_x = width // 2
    center_y = height // 2
    
    # XZ切片（垂直切片）
    xz_slice = volume_3d[:, center_y, :]
    # YZ切片（垂直切片）
    yz_slice = volume_3d[:, :, center_x]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(xz_slice, cmap='gray', aspect=aspect_ratio_xz)
    plt.title(f'XZ垂直切片 (y={center_y})')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(yz_slice, cmap='gray', aspect=aspect_ratio_yz)
    plt.title(f'YZ垂直切片 (x={center_x})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_vertical_slices.png", dpi=150)
    plt.close()
    
    print(f"已保存预览图像到: {preview_dir}")

def main():
    parser = argparse.ArgumentParser(description='将2D线条图像转换为3D结构')
    
    # 输入输出参数
    parser.add_argument('input', type=str, help='输入图像文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出MRC文件路径')
    parser.add_argument('--output-lefttop', type=str, help='左上区域输出MRC文件路径')
    parser.add_argument('--output-righttop', type=str, help='右上区域输出MRC文件路径')
    parser.add_argument('--output-bottom', type=str, help='底部区域输出MRC文件路径')
    
    # 线条提取参数
    parser.add_argument('--extraction-method', '-em', type=str, default='ridge',
                        choices=['canny', 'threshold', 'ridge'],
                        help='线条提取方法: canny, threshold, ridge')
    parser.add_argument('--threshold', '-th', type=float, default=0.05,
                        help='线条提取阈值')
    parser.add_argument('--line-width', '-lw', type=int, default=3,
                        help='线条宽度')
    
    # 3D转换参数
    parser.add_argument('--layers', '-l', type=int, default=21,
                        help='3D体积的层数，默认21')
    parser.add_argument('--z-distribution', '-zd', type=str, default='wave',
                        choices=['linear', 'gaussian', 'distance_transform', 'random', 'wave'],
                        help='Z方向分布方式: linear, gaussian, distance_transform, random, wave')
    parser.add_argument('--tilt-factor', '-tf', type=float, default=0.5,
                        help='控制Z方向变化的因子')
    parser.add_argument('--z-thickness', '-zt', type=int, default=5,
                        help='Z方向上每条线的厚度，默认5')
    parser.add_argument('--intensity-factor', '-if', type=float, default=1.5,
                        help='线条亮度因子')
    parser.add_argument('--z-blur', '-zb', type=float, default=2.0,
                        help='Z方向模糊程度 (值越大越模糊)，默认2.0')
    parser.add_argument('--random-factor', '-rf', type=float, default=1.0,
                        help='Z方向随机程度 (值越大越随机，仅用于random分布方式)')
    parser.add_argument('--tilt-preserve', '-tp', type=float, default=1.6,
                        help='倾斜保持系数 (0-1之间，值越大保留的倾斜趋势越明显，仅用于random分布方式)')
    parser.add_argument('--z-stretch', '-zs', type=float, default=4.0,
                        help='Z轴拉伸系数 (值越大Z轴拉伸效果越明显)，默认4.0')
    parser.add_argument('--wave-frequency', '-wf', type=float, default=2.0,
                        help='波浪频率 (值越大，波动越频繁，仅用于wave分布方式)')
    parser.add_argument('--wave-amplitude', '-wa', type=float, default=0.8,
                        help='波浪幅度 (0-1之间，值越大波动越剧烈，仅用于wave分布方式)')
    parser.add_argument('--use-fixed-seed', '-ufs', action='store_true',
                        help='使用固定随机种子以确保每次生成相同结果')
    parser.add_argument('--target-width', type=int, default=1403,
                        help='目标图像宽度，默认1403')
    parser.add_argument('--target-height', type=int, default=920,
                        help='目标图像高度，默认920')
    
    # 其他参数
    parser.add_argument('--preview', '-p', action='store_true',
                        help='生成预览图像')
    
    args = parser.parse_args()
    
    # 设置固定目标尺寸
    target_width = args.target_width
    target_height = args.target_height
    target_depth = args.layers
    
    # 读取输入图像
    print(f"读取图像: {args.input}")
    original_image = read_image(args.input)
    if original_image is None:
        print("无法读取输入图像，退出程序")
        return
    
    # 检查图像尺寸是否接近1004x1004
    orig_height, orig_width = original_image.shape[:2]
    print(f"原始图像尺寸: {orig_width}×{orig_height}")
    
    # 调整图像尺寸到1004x1004（如果需要）
    if orig_width != 1004 or orig_height != 1004:
        print(f"调整图像尺寸到 1004×1004...")
        original_image = cv2.resize(original_image, (1004, 1004), 
                                   interpolation=cv2.INTER_CUBIC)
        print(f"调整后图像尺寸: {original_image.shape[1]}×{original_image.shape[0]}")
    
    # 定义三个区域的裁剪坐标
    # 左上区域: 483x483，从(0,0)开始
    # 右上区域: 483x483，从(521,0)开始
    # 底部区域: 483x483，从(260,521)开始
    regions = {
        'lefttop': {'x': 0, 'y': 0, 'width': 483, 'height': 483, 'output': args.output_lefttop},
        'righttop': {'x': 521, 'y': 0, 'width': 483, 'height': 483, 'output': args.output_righttop},
        'bottom': {'x': 260, 'y': 521, 'width': 483, 'height': 483, 'output': args.output_bottom}
    }
    
    # 处理每个区域
    for region_name, region_info in regions.items():
        # 如果没有指定该区域的输出路径，则跳过
        if region_info['output'] is None and args.output is None:
            # 基于输入路径生成默认输出名称
            input_base = os.path.splitext(args.input)[0]
            region_info['output'] = f"{input_base}_3d_{region_name}_{target_depth}x{region_info['width']}x{region_info['height']}.mrc"
        elif region_info['output'] is None:
            # 如果没有指定该区域的输出，但指定了默认输出，则跳过该区域
            print(f"跳过 {region_name} 区域，因为未指定输出路径")
            continue
        
        print(f"\n处理 {region_name} 区域...")
        # 裁剪区域
        region_image = original_image[
            region_info['y']:region_info['y']+region_info['height'], 
            region_info['x']:region_info['x']+region_info['width']
        ]
        
        # 提取线条
        print(f"使用 {args.extraction_method} 方法提取线条...")
        lines_image = extract_lines(
            region_image, 
            method=args.extraction_method,
            threshold=args.threshold,
            line_width=args.line_width
        )
        
        # 保存线条图像以便检查（添加旋转和翻转）
        lines_output = os.path.splitext(region_info['output'])[0] + "_lines.png"
        # 旋转180度并左右镜像翻转
        lines_image_save = cv2.rotate(lines_image, cv2.ROTATE_180)
        lines_image_save = cv2.flip(lines_image_save, 1)  # 1表示左右翻转
        plt.imsave(lines_output, lines_image_save, cmap='gray')
        print(f"已保存线条图像到: {lines_output}")
        
        # 创建3D体积
        print(f"创建 {region_name} 区域的3D体积，使用 {args.z_distribution} 分布，层数 {target_depth}...")
        print("开始处理，这可能需要一些时间，请耐心等待...")
        
        # 计算总非零点数，用于进度估计
        non_zero_points = np.count_nonzero(lines_image)
        print(f"需要处理 {non_zero_points} 个线条点，生成 {target_depth} 层的3D体积")
        
        volume_3d = create_3d_from_lines(
            lines_image,
            num_layers=target_depth,
            z_distribution=args.z_distribution,
            tilt_factor=args.tilt_factor,
            z_thickness=args.z_thickness,
            intensity_factor=args.intensity_factor,
            z_blur=args.z_blur,
            random_factor=args.random_factor,
            tilt_preserve=args.tilt_preserve,
            z_stretch=args.z_stretch,
            wave_frequency=args.wave_frequency,
            wave_amplitude=args.wave_amplitude,
            use_fixed_seed=args.use_fixed_seed
        )
        
        # 检查并确认最终体积尺寸
        depth, height, width = volume_3d.shape
        print(f"生成的 {region_name} 区域3D体积尺寸: {depth}×{height}×{width}")
        
        # 保存MRC文件
        print(f"正在保存 {region_name} 区域3D体积到: {region_info['output']}")
        success = save_mrc(volume_3d, region_info['output'])
        
        # 生成预览图像
        if args.preview and success:
            print(f"生成 {region_name} 区域预览图像...")
            output_prefix = os.path.splitext(region_info['output'])[0]
            save_preview(region_image, lines_image, volume_3d, output_prefix)
        
        if success:
            print(f"{region_name} 区域处理完成! 已生成大小为 {volume_3d.shape[0]}×{volume_3d.shape[1]}×{volume_3d.shape[2]} 的3D体积。")
            print(f"输出文件路径: {region_info['output']}")
        else:
            print(f"{region_name} 区域处理过程中出现错误")
    
    # 如果指定了整体输出路径，也处理整个图像
    if args.output is not None:
        print("\n处理整个图像...")
        # 将原始图像调整为请求的目标尺寸
        if target_width != 1004 or target_height != 1004:
            print(f"调整图像尺寸从 1004×1004 到 {target_width}×{target_height}...")
            resized_image = cv2.resize(original_image, (target_width, target_height), 
                                     interpolation=cv2.INTER_CUBIC)
            print(f"调整后图像尺寸: {resized_image.shape[1]}×{resized_image.shape[0]}")
        else:
            resized_image = original_image
        
        # 提取线条
        print(f"使用 {args.extraction_method} 方法提取线条...")
        lines_image = extract_lines(
            resized_image, 
            method=args.extraction_method,
            threshold=args.threshold,
            line_width=args.line_width
        )
        
        # 保存线条图像以便检查（添加旋转和翻转）
        lines_output = os.path.splitext(args.output)[0] + "_lines.png"
        # 旋转180度并左右镜像翻转
        lines_image_save = cv2.rotate(lines_image, cv2.ROTATE_180)
        lines_image_save = cv2.flip(lines_image_save, 1)  # 1表示左右翻转
        plt.imsave(lines_output, lines_image_save, cmap='gray')
        print(f"已保存线条图像到: {lines_output}")
        
        # 创建3D体积
        print(f"创建整个图像的3D体积，使用 {args.z_distribution} 分布，层数 {target_depth}...")
        print("开始处理，这可能需要一些时间，请耐心等待...")
        
        # 计算总非零点数，用于进度估计
        non_zero_points = np.count_nonzero(lines_image)
        print(f"需要处理 {non_zero_points} 个线条点，生成 {target_depth} 层的3D体积")
        
        volume_3d = create_3d_from_lines(
            lines_image,
            num_layers=target_depth,
            z_distribution=args.z_distribution,
            tilt_factor=args.tilt_factor,
            z_thickness=args.z_thickness,
            intensity_factor=args.intensity_factor,
            z_blur=args.z_blur,
            random_factor=args.random_factor,
            tilt_preserve=args.tilt_preserve,
            z_stretch=args.z_stretch,
            wave_frequency=args.wave_frequency,
            wave_amplitude=args.wave_amplitude,
            use_fixed_seed=args.use_fixed_seed
        )
        
        # 检查并确认最终体积尺寸
        depth, height, width = volume_3d.shape
        print(f"生成的整体3D体积尺寸: {depth}×{height}×{width}")
        
        # 保存MRC文件
        print(f"正在保存整体3D体积到: {args.output}")
        success = save_mrc(volume_3d, args.output)
        
        # 生成预览图像
        if args.preview and success:
            print("生成整体预览图像...")
            output_prefix = os.path.splitext(args.output)[0]
            save_preview(resized_image, lines_image, volume_3d, output_prefix)
        
        if success:
            print(f"整体处理完成! 已生成大小为 {volume_3d.shape[0]}×{volume_3d.shape[1]}×{volume_3d.shape[2]} 的3D体积。")
            print(f"输出文件路径: {args.output}")
        else:
            print("整体处理过程中出现错误")

if __name__ == "__main__":
    main() 