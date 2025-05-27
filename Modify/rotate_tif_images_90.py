#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm

def rotate_tif_images_clockwise_90_and_fliplr(folder_path, recursive=True):
    """
    顺时针旋转TIF图像90度，然后进行左右翻转。
    
    Args:
        folder_path (str): 包含TIF图像的文件夹路径
        recursive (bool): 是否递归处理子目录
    """
    # 转换为Path对象以便更好地处理路径
    folder = Path(folder_path)
    
    # 获取所有TIF文件（包括子目录中的文件，如果recursive为True）
    pattern = "**/*.tif" if recursive else "*.tif"
    tif_files = list(folder.glob(pattern))
    
    print(f"在 {folder_path}" + ("及其子目录" if recursive else "") + f"中找到 {len(tif_files)} 个TIF文件")
    
    # 使用tqdm显示进度条
    for tif_file in tqdm(tif_files, desc="处理TIF文件"):
        try:
            # 使用scikit-image读取图像
            img = io.imread(tif_file)
            
            # 检查图像是否为3D（图像堆栈）
            if len(img.shape) == 3:
                # 处理堆栈中的每一帧
                processed_img = np.zeros_like(img)
                for i in range(img.shape[0]):
                    # 顺时针旋转90度 (k=3相当于顺时针旋转90度，因为k=1是逆时针90度)
                    rotated = np.rot90(img[i], k=3)
                    # 左右翻转
                    processed_img[i] = np.fliplr(rotated)
            else:
                # 单个2D图像
                # 顺时针旋转90度
                rotated = np.rot90(img, k=3)
                # 左右翻转
                processed_img = np.fliplr(rotated)
            
            # 保存处理后的图像，覆盖原文件
            io.imsave(tif_file, processed_img)
            
        except Exception as e:
            print(f"处理 {tif_file.name} 时出错: {str(e)}")

def rotate_tif_images_counterclockwise_90_batch(folder_path, recursive=True):
    """
    批量处理：逆时针旋转TIF图像90度。
    
    Args:
        folder_path (str): 包含TIF图像的文件夹路径
        recursive (bool): 是否递归处理子目录
    """
    # 转换为Path对象以便更好地处理路径
    folder = Path(folder_path)
    
    # 获取所有TIF文件（包括子目录中的文件，如果recursive为True）
    pattern = "**/*.tif" if recursive else "*.tif"
    tif_files = list(folder.glob(pattern))
    
    print(f"在 {folder_path}" + ("及其子目录" if recursive else "") + f"中找到 {len(tif_files)} 个TIF文件")
    
    # 使用tqdm显示进度条
    for tif_file in tqdm(tif_files, desc="处理TIF文件"):
        try:
            # 使用scikit-image读取图像
            img = io.imread(tif_file)
            
            # 检查图像是否为3D（图像堆栈）
            if len(img.shape) == 3:
                # 处理堆栈中的每一帧
                processed_img = np.zeros_like(img)
                for i in range(img.shape[0]):
                    # 逆时针旋转90度 (k=1表示逆时针旋转90度)
                    processed_img[i] = np.rot90(img[i], k=1)
            else:
                # 单个2D图像
                # 逆时针旋转90度
                processed_img = np.rot90(img, k=1)
            
            # 保存处理后的图像，覆盖原文件
            io.imsave(tif_file, processed_img)
            
        except Exception as e:
            print(f"处理 {tif_file.name} 时出错: {str(e)}")

def rotate_tif_images_counterclockwise_90(input_file, output_file, angle=90):
    """
    旋转单个TIF图像并保存到指定路径。
    
    Args:
        input_file (str): 输入TIF图像的文件路径
        output_file (str): 输出TIF图像的文件路径
        angle (int): 旋转角度，默认为90度。必须是90的倍数，如90、180、270或-90等
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 计算旋转次数(k值)
        # 逆时针旋转90度对应k=1
        # 旋转180度对应k=2
        # 旋转270度对应k=3
        k = (angle % 360) // 90
        if k < 0:
            k += 4  # 负角度调整
            
        print(f"旋转图像 {input_file} ，角度 {angle}° (k={k})，输出到 {output_file}")
        
        # 使用scikit-image读取图像
        img = io.imread(input_file)
        
        # 检查图像是否为3D（图像堆栈）
        if len(img.shape) == 3:
            # 处理堆栈中的每一帧
            processed_img = np.zeros_like(img)
            for i in range(img.shape[0]):
                processed_img[i] = np.rot90(img[i], k=k)
        else:
            # 单个2D图像
            processed_img = np.rot90(img, k=k)
        
        # 保存处理后的图像
        io.imsave(output_file, processed_img)
        print(f"成功将 {input_file} 旋转 {angle}° 并保存到 {output_file}")
        
    except Exception as e:
        print(f"处理 {os.path.basename(input_file)} 时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # TIF图像所在文件夹的路径
    folder_path = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\Microtubules\Microtubules(OK)\LF_Mic"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
    else:
        print(f"开始处理 '{folder_path}' 中的TIF文件...")
        rotate_tif_images_counterclockwise_90_batch(folder_path, recursive=True)
        print("处理完成！") 