#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import subprocess
import glob
import argparse
import shutil

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def main():
    parser = argparse.ArgumentParser(description='批量处理MRC文件并生成3D数据')
    
    # 参数设置
    parser.add_argument('--input-base', '-i', type=str, default='D:\桌面\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\ER(raw)\ER',
                       help='输入基础路径，包含Cell_001到Cell_068的文件夹')
    parser.add_argument('--output-dir', '-o', type=str, default='生成数据集/ER_3D',
                       help='输出目录路径')
    parser.add_argument('--extraction-method', '-em', type=str, default='ridge',
                       help='线条提取方法')
    parser.add_argument('--z-distribution', '-zd', type=str, default='wave',
                       help='Z方向分布方式')
    parser.add_argument('--wave-frequency', '-wf', type=float, default=2.0,
                       help='波浪频率')
    parser.add_argument('--wave-amplitude', '-wa', type=float, default=0.8,
                       help='波浪幅度')
    parser.add_argument('--tilt-factor', '-tf', type=float, default=0.3,
                       help='倾斜因子')
    parser.add_argument('--z-thickness', '-zt', type=int, default=1,
                       help='Z方向厚度')
    parser.add_argument('--z-blur', '-zb', type=float, default=1.5,
                       help='Z方向模糊程度')
    parser.add_argument('--z-stretch', '-zs', type=float, default=1.0,
                       help='Z轴拉伸系数')
    parser.add_argument('--layers', '-l', type=int, default=21,
                       help='3D体积的层数，默认为838')
    parser.add_argument('--target-width', '-tw', type=int, default=483
                       help='目标图像宽度，默认1403')
    parser.add_argument('--target-height', '-th', type=int, default=483,
                       help='目标图像高度，默认920')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='为每个文件生成预览图像')
    parser.add_argument('--start-cell', '-sc', type=int, default=1,
                       help='起始Cell编号')
    parser.add_argument('--end-cell', '-ec', type=int, default=68,
                       help='结束Cell编号')
    
    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = args.output_dir
    ensure_dir_exists(output_dir)
    
    # 处理每个Cell文件夹
    processed_count = 0  # 成功处理的Cell数量
    processed_files = 0  # 成功处理的文件数量
    failed_cells = []    # 处理失败的Cell
    
    for cell_num in range(args.start_cell, args.end_cell + 1):
        cell_name = f"Cell_{cell_num:03d}"
        print(f"\n处理 {cell_name}...")
        
        # 构建GTSIM文件夹路径
        gtsim_dir = os.path.join(args.input_base, cell_name, "GTSIM")
        
        # 检查GTSIM文件夹是否存在
        if not os.path.exists(gtsim_dir):
            print(f"警告: {gtsim_dir} 不存在，跳过。")
            failed_cells.append(cell_name + " (文件夹不存在)")
            continue
        
        # 查找所有GTSIM_level_*.mrc文件
        mrc_pattern = os.path.join(gtsim_dir, "GTSIM_level_*.mrc")
        mrc_files = [f for f in glob.glob(mrc_pattern) if os.path.isfile(f) and 
                    os.path.basename(f).startswith("GTSIM_level_") and 
                    os.path.basename(f).endswith(".mrc") and
                    "_3d" not in os.path.basename(f)]  # 排除已生成的3D文件
        
        if not mrc_files:
            print(f"警告: 在 {gtsim_dir} 中没有找到MRC文件，跳过。")
            failed_cells.append(cell_name + " (没有找到MRC文件)")
            continue
        
        # 随机选择两个MRC文件
        num_files_to_select = min(2, len(mrc_files))  # 确保文件数量足够
        selected_mrcs = random.sample(mrc_files, num_files_to_select)
        print(f"随机选择了 {num_files_to_select} 个文件进行处理")
        
        # 为每个Cell创建输出子目录
        cell_output_dir = os.path.join(output_dir, cell_name)
        ensure_dir_exists(cell_output_dir)
        
        cell_processed = 0
        
        # 处理选择的每个MRC文件
        for selected_mrc in selected_mrcs:
            level_name = os.path.basename(selected_mrc).replace(".mrc", "")
            print(f"\n处理文件: {selected_mrc}")
            
            # 构建输出文件路径
            output_base = os.path.join(cell_output_dir, level_name)
            output_mrc = output_base + f"_3d_{args.layers}x{args.target_width}x{args.target_height}.mrc"
            
            # 构建命令
            cmd = [r"D:\桌面\科研\光学成像\hylfm-net\.venv\Scripts\python", r"D:\桌面\科研\光学成像\hylfm-net\处理代码\extract_lines_to_3d.py", 
                  selected_mrc,
                  "--output", output_mrc,
                  "--extraction-method", args.extraction_method,
                  "--z-distribution", args.z_distribution,
                  "--layers", str(args.layers),
                  "--z-stretch", str(args.z_stretch),
                  "--wave-frequency", str(args.wave_frequency),
                  "--wave-amplitude", str(args.wave_amplitude),
                  "--tilt-factor", str(args.tilt_factor),
                  "--z-thickness", str(args.z_thickness),
                  "--z-blur", str(args.z_blur),
                  "--target-width", str(args.target_width),
                  "--target-height", str(args.target_height)]
            
            if args.preview:
                cmd.append("--preview")
            
            # 执行命令
            print("执行命令:", " ".join(cmd))
            try:
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cell_processed += 1
                print(f"成功处理文件: {os.path.basename(selected_mrc)}")
                
                # 将预览图像也复制到输出目录
                if args.preview:
                    preview_pattern = os.path.splitext(output_mrc)[0] + "_*.png"
                    preview_files = glob.glob(preview_pattern)
                    if not preview_files:
                        # 如果没有找到以输出文件名为基础的预览图，则尝试查找原始文件生成的预览图
                        src_preview_files = glob.glob(os.path.join(os.path.dirname(selected_mrc), f"{level_name}_3d_*.png"))
                        
                        # 只复制源目录中的预览图像
                        for preview_file in src_preview_files:
                            if os.path.exists(preview_file):
                                target_file = os.path.join(cell_output_dir, os.path.basename(preview_file))
                                # 确保源和目标不是同一个文件
                                if os.path.abspath(preview_file) != os.path.abspath(target_file):
                                    shutil.copy2(preview_file, cell_output_dir)
                                    print(f"复制预览图像到 {cell_output_dir}: {os.path.basename(preview_file)}")
                    else:
                        print(f"预览图像已生成在输出目录: {len(preview_files)} 个文件")
                    
            except subprocess.CalledProcessError as e:
                print(f"处理文件 {os.path.basename(selected_mrc)} 时出错:")
                try:
                    stderr_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
                    print(f"错误信息: {stderr_output}")
                    failed_cells.append(f"{cell_name} - {os.path.basename(selected_mrc)} (处理失败)")
                except Exception:
                    print(f"无法解码错误信息")
                    failed_cells.append(f"{cell_name} - {os.path.basename(selected_mrc)} (处理失败)")
        
        if cell_processed > 0:
            processed_count += 1
            processed_files += cell_processed
            print(f"Cell {cell_name} 处理完成，成功处理了 {cell_processed} 个文件")
    
    # 打印汇总信息
    print("\n批处理完成!")
    print(f"成功处理: {processed_count} / {args.end_cell - args.start_cell + 1} 个Cell")
    print(f"成功处理文件: {processed_files} 个文件")
    
    if failed_cells:
        print(f"失败的Cell ({len(failed_cells)}):")
        for cell in failed_cells:
            print(f"  - {cell}")

if __name__ == "__main__":
    main() 