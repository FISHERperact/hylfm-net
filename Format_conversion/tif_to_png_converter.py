#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path
from skimage import io, img_as_ubyte
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def convert_tif_to_png(input_folder, output_folder, recursive=True):
    """
    将指定目录下的所有TIF文件转换为PNG格式，并保存到统一的输出目录。
    
    Args:
        input_folder (str): 包含TIF文件的文件夹路径
        output_folder (str): 输出PNG文件的文件夹路径
        recursive (bool): 是否递归处理子目录
    """
    # 转换为Path对象以便更好地处理路径
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # 创建输出目录（如果不存在）
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"创建输出目录: {output_folder}")
    
    # 获取所有TIF文件（包括子目录中的文件，如果recursive为True）
    pattern = "**/*.tif" if recursive else "*.tif"
    tif_files = list(input_path.glob(pattern))
    
    print(f"在 {input_folder}" + ("及其子目录" if recursive else "") + f"中找到 {len(tif_files)} 个TIF文件")
    
    # 用于记录已处理的文件名，避免重复
    processed_filenames = set()
    
    # 使用tqdm显示进度条
    for tif_file in tqdm(tif_files, desc="转换TIF到PNG"):
        try:
            # 使用scikit-image读取图像
            img = io.imread(tif_file)
            
            # 相对路径用于构建输出文件名，防止重名
            rel_path = tif_file.relative_to(input_path)
            rel_dir = str(rel_path.parent).replace('\\', '_').replace('/', '_')
            
            # 构建输出文件名
            if rel_dir and rel_dir != '.':
                base_name = f"{rel_dir}_{tif_file.stem}"
            else:
                base_name = tif_file.stem
            
            # 检查是否需要处理3D图像
            if len(img.shape) == 3 and img.shape[0] > 1:
                # 处理3D图像，提取中间层
                z_dim = img.shape[0]
                middle_layer = z_dim // 2
                
                # 如果是多层图像，保存中间层作为PNG
                layer = img[middle_layer]
                
                # 如果层级太多，可能会导致多个文件具有相同中间层编号，加上文件名以防止冲突
                png_file_name = f"{base_name}_middle_layer.png"
                png_path = output_path / png_file_name
                
                # 检查文件名是否已存在，如果存在则添加序号
                counter = 1
                while png_file_name in processed_filenames:
                    png_file_name = f"{base_name}_middle_layer_{counter}.png"
                    png_path = output_path / png_file_name
                    counter += 1
                
                processed_filenames.add(png_file_name)
                
                # 归一化处理
                if layer.dtype != np.uint8:
                    layer_min = np.min(layer)
                    layer_max = np.max(layer)
                    if layer_max > layer_min:
                        layer = (layer - layer_min) / (layer_max - layer_min)
                    layer = img_as_ubyte(layer)
                
                # 保存为PNG
                plt.imsave(png_path, layer, cmap='gray')
                print(f"保存3D图像 {tif_file.name} 的中间层 ({middle_layer}) 到 {png_path}")
                
            else:
                # 处理2D图像
                # 构建PNG文件路径
                png_file_name = f"{base_name}.png"
                png_path = output_path / png_file_name
                
                # 检查文件名是否已存在，如果存在则添加序号
                counter = 1
                while png_file_name in processed_filenames:
                    png_file_name = f"{base_name}_{counter}.png"
                    png_path = output_path / png_file_name
                    counter += 1
                
                processed_filenames.add(png_file_name)
                
                # 归一化处理
                if len(img.shape) == 2 and img.dtype != np.uint8:
                    img_min = np.min(img)
                    img_max = np.max(img)
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    img = img_as_ubyte(img)
                
                # 保存为PNG
                if len(img.shape) == 2:
                    plt.imsave(png_path, img, cmap='gray')
                else:
                    plt.imsave(png_path, img)
                
                print(f"保存图像 {tif_file.name} 到 {png_path}")
            
        except Exception as e:
            print(f"处理 {tif_file.name} 时出错: {str(e)}")
    
    print(f"转换完成！共转换 {len(processed_filenames)} 个PNG文件到 {output_folder}")

if __name__ == "__main__":
    # 定义输入和输出路径
    input_folder = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\ER"
    output_folder = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\ER_PNG"
    
    # 调用函数转换TIF到PNG
    convert_tif_to_png(input_folder, output_folder, recursive=True) 