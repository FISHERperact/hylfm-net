import os
import numpy as np
from scipy import io
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc  # 垃圾回收模块，用于释放内存

def mat_to_tif(mat_file_path, output_dir=None):
    """
    将单个MAT文件转换为TIF格式。
    
    Args:
        mat_file_path (str): MAT文件路径
        output_dir (str, optional): 输出目录，默认与MAT文件相同目录
        
    Returns:
        str: 保存的TIF文件路径
    """
    # 加载MAT文件
    try:
        # 先检查输出文件是否已存在，避免重复处理
        mat_path = Path(mat_file_path)
        if output_dir is None:
            output_dir = mat_path.parent
        else:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
        
        tif_path = output_dir / f"{mat_path.stem}.tif"
        
        if tif_path.exists():
            print(f"文件 {tif_path.name} 已存在，跳过处理")
            return str(tif_path)
        
        # 加载MAT文件内容
        print(f"加载MAT文件: {os.path.basename(mat_file_path)}")
        mat_data = io.loadmat(mat_file_path)
        
        # 打印MAT文件中的变量，帮助调试
        print(f"MAT文件 {os.path.basename(mat_file_path)} 包含以下变量:")
        for key in mat_data.keys():
            if not key.startswith('__'):  # 跳过内部变量
                print(f"  - {key}: shape {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
        
        # 尝试找到主要的图像数据
        image_var = None
        
        # 常见的图像变量名
        possible_vars = ['volume', 'img', 'image', 'data', 'vol', 'stack', 'rec']
        
        # 首先尝试通过变量名找到图像数据
        for var in possible_vars:
            if var in mat_data:
                image_var = var
                break
        
        # 如果通过名称找不到，则尝试找到最大的数组变量
        if image_var is None:
            max_size = 0
            for key in mat_data.keys():
                if not key.startswith('__'):
                    if hasattr(mat_data[key], 'size') and mat_data[key].size > max_size:
                        max_size = mat_data[key].size
                        image_var = key
        
        if image_var is None:
            raise ValueError(f"无法在MAT文件中找到图像数据")
        
        # 获取图像数据
        print(f"使用变量 '{image_var}' 作为图像数据")
        image_data = mat_data[image_var].copy()  # 创建副本以允许释放原始数据
        
        # 释放原始MAT数据以节省内存
        del mat_data
        gc.collect()
        
        # 处理数据，确保是适合TIF格式的数据类型和规模
        # 如果是复数数据，取绝对值
        if np.iscomplexobj(image_data):
            print(f"  * 将复数数据转换为绝对值")
            image_data = np.abs(image_data)
        
        # 确保数据为3D或2D
        if image_data.ndim > 3:
            print(f"  * 将{image_data.ndim}维数据转换为3D (取第一个通道/时间点)")
            # 如果是4D或更高维，尝试保留前三维
            while image_data.ndim > 3:
                image_data = image_data[0]
        
        # 归一化图像数据到0-1范围
        data_min = np.min(image_data)
        data_max = np.max(image_data)
        
        if data_max > data_min:
            print(f"  * 归一化数据范围: [{data_min}, {data_max}] -> [0, 1]")
            image_data = (image_data - data_min) / (data_max - data_min)
        
        # 将数据缩放到0-65535 (16位TIF)
        image_data = (image_data * 65535).astype(np.uint16)
        
        # 保存为TIF文件
        print(f"保存为TIF文件: {tif_path}")
        tifffile.imwrite(tif_path, image_data)
        
        # 释放图像数据内存
        del image_data
        gc.collect()
        
        print(f"成功将 {mat_path.name} 转换为 {tif_path.name}")
        
        # 可选地创建中心切片的PNG预览，默认关闭以加快处理速度
        create_preview = False  # 设置为True以启用预览生成
        
        if create_preview and tif_path.exists():
            try:
                # 重新加载TIF文件以生成预览
                preview_data = tifffile.imread(str(tif_path))
                
                if preview_data.ndim == 3:
                    preview_dir = output_dir / "previews"
                    if not preview_dir.exists():
                        preview_dir.mkdir(parents=True)
                    
                    # 创建中心切片的预览图
                    center_slice = preview_data.shape[0] // 2
                    plt.figure(figsize=(10, 8))
                    plt.imshow(preview_data[center_slice], cmap='gray')
                    plt.title(f"{mat_path.stem} - 中心切片 ({center_slice})")
                    plt.colorbar()
                    
                    preview_path = preview_dir / f"{mat_path.stem}_preview.png"
                    plt.savefig(preview_path)
                    plt.close()
                    
                    print(f"保存预览图到 {preview_path}")
                
                # 释放预览数据内存
                del preview_data
                gc.collect()
            except Exception as preview_error:
                print(f"生成预览图时出错: {str(preview_error)}")
        
        return str(tif_path)
    
    except Exception as e:
        print(f"处理文件 {mat_file_path} 时发生错误: {str(e)}")
        return None

def convert_all_mat_files(input_dir, output_dir=None):
    """
    转换指定目录下的所有MAT文件为TIF格式
    
    Args:
        input_dir (str): 包含MAT文件的目录
        output_dir (str, optional): 保存TIF文件的目录，默认与输入目录相同
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"错误: 目录 {input_dir} 不存在")
        return
    
    # 如果未指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = input_path
    else:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
    
    # 获取所有MAT文件
    mat_files = list(input_path.glob("*.mat"))
    
    if not mat_files:
        print(f"警告: 在 {input_dir} 中没有找到MAT文件")
        return
    
    print(f"在 {input_dir} 中找到 {len(mat_files)} 个MAT文件")
    
    # 转换每个MAT文件
    converted_files = []
    failed_files = []
    
    # 使用tqdm添加进度条
    for mat_file in tqdm(mat_files, desc="转换MAT文件"):
        print(f"\n处理文件: {mat_file.name}")
        try:
            tif_path = mat_to_tif(str(mat_file), output_dir)
            if tif_path:
                converted_files.append(tif_path)
            else:
                failed_files.append(str(mat_file))
            
            # 手动触发垃圾回收，释放内存
            gc.collect()
            
        except Exception as e:
            print(f"转换 {mat_file.name} 时发生错误: {str(e)}")
            failed_files.append(str(mat_file))
    
    # 打印总结
    print("\n转换完成!")
    print(f"成功转换: {len(converted_files)}/{len(mat_files)} 个文件")
    
    if failed_files:
        print(f"失败转换: {len(failed_files)}/{len(mat_files)} 个文件:")
        for fail in failed_files:
            print(f"  - {os.path.basename(fail)}")
    
    return converted_files, failed_files

if __name__ == "__main__":
    # 定义输入目录
    input_directory = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\Microtubules\Microtubules(OK)\mat"
    
    # 创建一个专门的输出目录用于存放TIF文件
    output_directory = r"F:\科研\光学成像\hylfm-net\光场仿真数据集\自拟数据集\Microtubules\Microtubules(OK)\tif"
    
    # 转换所有MAT文件
    convert_all_mat_files(input_directory, output_directory) 