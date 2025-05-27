import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

def sort_tif_files(folder_path):
    """
    读取指定文件夹中的TIF文件，并按照Cell编号和特定顺序排列。
    对于每个Cell，顺序为: wfm, lefttop_wfm, righttop_wfm
    
    Args:
        folder_path (str): 包含TIF文件的文件夹路径
    
    Returns:
        list: 按照特定顺序排序的文件路径列表
    """
    # 转换为Path对象以便更好地处理路径
    folder = Path(folder_path)
    
    # 获取文件夹中的所有TIF文件
    tif_files = list(folder.glob("*.tif"))
    
    print(f"在 {folder_path} 中找到 {len(tif_files)} 个TIF文件")
    
    # 使用正则表达式从文件名中提取Cell编号
    cell_pattern = re.compile(r'ER_3D_Cell_(\d+)_GTSIM')
    
    # 创建一个字典，用于按Cell编号分组文件
    cell_groups = defaultdict(list)
    
    # 遍历所有TIF文件并按Cell编号分组
    for tif_file in tif_files:
        match = cell_pattern.search(tif_file.name)
        if match:
            cell_num = int(match.group(1))
            cell_groups[cell_num].append(tif_file)
    
    # 最终排序的文件列表
    sorted_files = []
    
    # 按Cell编号排序（从001到068）
    for cell_num in sorted(cell_groups.keys()):
        files = cell_groups[cell_num]
        
        # 对于每个Cell，按指定顺序排列文件：wfm, lefttop_wfm, righttop_wfm
        wfm_file = None
        lefttop_file = None
        righttop_file = None
        
        for file in files:
            if "lefttop_wfm" in file.name:
                lefttop_file = file
            elif "righttop_wfm" in file.name:
                righttop_file = file
            elif "_wfm.tif" in file.name:  # 标准wfm文件名以_wfm.tif结尾
                wfm_file = file
        
        # 按指定顺序添加文件（如果存在）
        if wfm_file:
            sorted_files.append(wfm_file)
        if lefttop_file:
            sorted_files.append(lefttop_file)
        if righttop_file:
            sorted_files.append(righttop_file)
    
    return sorted_files

def print_sorted_files(files):
    """打印排序后的文件列表"""
    print("\n按排序顺序列出的文件：")
    print("-" * 50)
    for i, file in enumerate(files):
        print(f"{i+1:3d}. {file.name}")
    print("-" * 50)
    print(f"总共 {len(files)} 个文件")

def copy_files_to_new_folder(sorted_files, output_folder):
    """
    将排序后的文件复制到新文件夹中，通过添加数字前缀确保文件按照指定顺序排列
    
    Args:
        sorted_files (list): 排序后的文件路径列表
        output_folder (str): 目标文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    output_path = Path(output_folder)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"创建输出文件夹：{output_folder}")
    
    # 复制文件到输出文件夹，并添加序号前缀
    print(f"\n开始复制文件到 {output_folder}...")
    
    # 计算需要的前缀位数
    total_files = len(sorted_files)
    prefix_digits = len(str(total_files))
    
    # 记录新文件名与原文件名的映射，用于打印信息
    filename_mapping = []
    
    # 复制文件并添加序号前缀
    for i, src_file in enumerate(sorted_files):
        # 生成序号前缀
        prefix = f"{i+1:0{prefix_digits}d}_"
        
        # 构建目标文件路径（带前缀）
        new_filename = f"{prefix}{src_file.name}"
        dst_file = output_path / new_filename
        
        # 复制文件
        shutil.copy2(src_file, dst_file)
        
        # 记录文件名映射
        filename_mapping.append((src_file.name, new_filename))
        
        # 每复制10个文件打印一次进度
        if (i + 1) % 10 == 0 or i == len(sorted_files) - 1:
            print(f"已复制 {i+1}/{len(sorted_files)} 个文件")
    
    print(f"所有文件已成功复制到 {output_folder}")
    
    # 将文件名映射保存到文本文件
    mapping_file = "filename_mapping.txt"
    with open(mapping_file, "w", encoding="utf-8") as f:
        f.write("原文件名 -> 新文件名\n")
        f.write("-" * 50 + "\n")
        for original, new_name in filename_mapping:
            f.write(f"{original} -> {new_name}\n")
    
    print(f"文件名映射已保存到 {mapping_file}")

def main():
    # 包含TIF文件的文件夹路径
    folder_path = "宽场数据(mix)"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在。")
        return
    
    # 获取排序后的文件列表
    sorted_files = sort_tif_files(folder_path)
    
    # 打印排序后的文件列表
    print_sorted_files(sorted_files)
    
    # 将排序后的文件列表保存到文本文件中
    output_file = "sorted_tif_files.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for file in sorted_files:
            f.write(f"{file.name}\n")
    
    print(f"排序后的文件列表已保存到 {output_file}")
    
    # 创建新文件夹并复制文件
    output_folder = "宽场数据_已排序_带序号"
    copy_files_to_new_folder(sorted_files, output_folder)

if __name__ == "__main__":
    main() 