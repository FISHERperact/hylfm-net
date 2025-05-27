#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import mrcfile
import numpy as np
from tqdm import tqdm
import traceback
import argparse

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def crop_mrc_file(input_file, output_file, target_width=483, target_height=483):
    """
    裁剪MRC文件到指定宽度和高度，保持深度不变
    
    参数:
    - input_file: 输入MRC文件路径
    - output_file: 输出MRC文件路径
    - target_width: 目标宽度，默认483
    - target_height: 目标高度，默认483
    
    返回:
    - 是否成功裁剪
    """
    try:
        # 打开MRC文件，允许不标准的MRC文件
        with mrcfile.open(input_file, permissive=True, mode='r') as mrc:
            # 获取原始数据
            data = mrc.data.copy()  # 复制数据以防止内存问题
            
            # 获取原始尺寸
            depth, height, width = data.shape
            
            # 计算开始位置（居中裁剪）
            start_h = max(0, (height - target_height) // 2)
            start_w = max(0, (width - target_width) // 2)
            
            # 计算结束位置
            end_h = min(height, start_h + target_height)
            end_w = min(width, start_w + target_width)
            
            # 检查原始尺寸是否小于目标尺寸
            if height < target_height or width < target_width:
                print(f"警告: {os.path.basename(input_file)} 的尺寸 ({width}x{height}) 小于目标尺寸 ({target_width}x{target_height})，将进行填充而不是裁剪")
                # 创建新的空数组（用0填充）
                new_data = np.zeros((depth, target_height, target_width), dtype=data.dtype)
                # 计算居中的位置
                pad_start_h = max(0, (target_height - height) // 2)
                pad_start_w = max(0, (target_width - width) // 2)
                # 将原数据复制到新数组
                new_data[:, pad_start_h:pad_start_h+height, pad_start_w:pad_start_w+width] = data
            else:
                # 裁剪数据
                new_data = data[:, start_h:end_h, start_w:end_w]
                
                # 如果裁剪后的尺寸不匹配目标尺寸，调整大小
                if new_data.shape[1] != target_height or new_data.shape[2] != target_width:
                    pad_h = target_height - new_data.shape[1]
                    pad_w = target_width - new_data.shape[2]
                    pad_h_top = pad_h // 2
                    pad_h_bottom = pad_h - pad_h_top
                    pad_w_left = pad_w // 2
                    pad_w_right = pad_w - pad_w_left
                    new_data = np.pad(new_data, ((0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), 'constant')
            
            # 创建新的MRC文件
            with mrcfile.new(output_file, overwrite=True) as new_mrc:
                new_mrc.set_data(new_data)
                # 复制原始文件的头信息（避免直接赋值可能导致的属性错误）
                # 不再尝试直接复制header和extended_header
                # 而是使用update_header_from_data让mrcfile库自动设置正确的头信息
                new_mrc.update_header_from_data()
                
                # 可选：复制一些额外的元数据（如果需要的话）
                if hasattr(mrc.header, 'origin') and hasattr(new_mrc.header, 'origin'):
                    new_mrc.header.origin = mrc.header.origin
                if hasattr(mrc.header, 'map') and hasattr(new_mrc.header, 'map'):
                    new_mrc.header.map = mrc.header.map
            
            return True
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        print(f"详细错误信息: {error_details}")
        return False

def validate_output_file(file_path, expected_width, expected_height):
    """验证输出文件是否存在且有正确的尺寸"""
    try:
        if not os.path.exists(file_path):
            return False, "文件不存在"
            
        # 检查文件大小是否大于0
        if os.path.getsize(file_path) == 0:
            return False, "文件大小为0"
            
        # 尝试打开文件并检查尺寸
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data
            depth, height, width = data.shape
            
            if width != expected_width or height != expected_height:
                return False, f"尺寸不匹配 - 预期 {expected_width}x{expected_height}，实际 {width}x{height}"
                
            return True, "验证通过" 
            
    except Exception as e:
        return False, f"验证出错: {str(e)}"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='裁剪MRC文件到指定宽度和高度，保持深度不变')
    parser.add_argument('--input-dir', '-i', type=str, 
                       default=r"D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\生成数据集（2.0）",
                       help='输入目录路径')
    parser.add_argument('--output-dir', '-o', type=str, 
                       default=r"D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\生成数据集（3.0）",
                       help='输出目录路径')
    parser.add_argument('--width', '-w', type=int, default=483,
                       help='目标宽度')
    parser.add_argument('--height', '-ht', type=int, default=483,
                       help='目标高度')
    args = parser.parse_args()
    
    # 获取参数
    input_dir = args.input_dir
    output_dir = args.output_dir
    target_width = args.width
    target_height = args.height
    
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    # 查找所有MRC文件
    mrc_files = glob.glob(os.path.join(input_dir, "**", "*.mrc"), recursive=True)
    
    if not mrc_files:
        print(f"在 {input_dir} 中没有找到MRC文件")
        return
    
    print(f"找到 {len(mrc_files)} 个MRC文件")
    
    # 处理计数
    success_count = 0
    failed_count = 0
    
    # 处理每个MRC文件
    for input_file in tqdm(mrc_files, desc="裁剪MRC文件"):
        # 构建输出文件路径
        rel_path = os.path.relpath(input_file, input_dir)
        output_file = os.path.join(output_dir, rel_path)
        
        # 确保输出目录存在
        output_subdir = os.path.dirname(output_file)
        ensure_dir_exists(output_subdir)
        
        # 裁剪MRC文件
        if crop_mrc_file(input_file, output_file, target_width, target_height):
            # 验证输出文件
            is_valid, validation_message = validate_output_file(output_file, target_width, target_height)
            if is_valid:
                success_count += 1
                print(f"成功处理: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
            else:
                failed_count += 1
                print(f"处理文件 {input_file} 验证失败: {validation_message}")
                # 尝试删除无效文件
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                        print(f"已删除无效文件: {output_file}")
                except:
                    pass
        else:
            failed_count += 1
    
    # 打印处理结果
    print("\n裁剪完成!")
    print(f"成功裁剪: {success_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main() 