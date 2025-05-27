import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging
from scipy import stats
from skimage import exposure
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from matplotlib.font_manager import FontProperties

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 配置matplotlib字体
try:
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
    plt.rcParams['font.family'] = font.get_name()
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TifAnalyzer:
    """3D TIF数据分析和处理类"""
    
    def __init__(self):
        """初始化分析器"""
        self.expected_depth = 21  # 预期的图像深度
        
        # 质量评估阈值
        self.thresholds = {
            'min_intensity': 0.001,    # 最小平均强度（归一化后）
            'max_intensity': 0.999,    # 最大平均强度（归一化后）
            'min_variance': 0.0001,    # 最小方差
            'min_snr': 1.0,           # 最小信噪比
            'min_entropy': 0.1,        # 最小结构熵
            'overexposure_ratio': 0.4  # 过曝区域占比阈值
        }
        
        # 存储统计数据
        self.stats_data = {
            'mean_intensities': [],
            'variances': [],
            'snrs': [],
            'entropies': [],
            'filenames': [],
            'anomalies': []  # 存储异常样本信息
        }
    
    def calculate_stats(self, image: np.ndarray) -> Dict[str, float]:
        """计算图像统计指标
        
        Args:
            image: 3D图像数据
            
        Returns:
            包含统计指标的字典
        """
        # 归一化到[0,1]范围
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # 计算基本统计量
        mean_intensity = np.mean(image_norm)
        variance = np.var(image_norm)
        
        # 计算信噪比
        signal = np.mean(image_norm)
        noise = np.std(image_norm)
        snr = signal / (noise + 1e-8)
        
        # 计算结构熵
        hist, _ = np.histogram(image_norm.flatten(), bins=256, range=(0, 1))
        hist_norm = hist / float(hist.sum() + 1e-8)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        
        # 计算过曝区域占比
        overexposed_ratio = np.mean(image_norm > 0.95)
        
        return {
            'mean_intensity': mean_intensity,
            'variance': variance,
            'snr': snr,
            'entropy': entropy,
            'overexposed_ratio': overexposed_ratio
        }
    
    def check_anomaly(self, image: np.ndarray, stats: Dict[str, float], filename: str) -> Tuple[bool, str]:
        """检查图像是否异常
        
        Args:
            image: 3D图像数据
            stats: 统计指标
            filename: 文件名
            
        Returns:
            (是否异常, 异常原因)
        """
        # 检查深度
        if image.shape[0] != self.expected_depth:
            return True, f"深度异常(期望{self.expected_depth}, 实际{image.shape[0]})"
        
        # 检查是否为纯黑图像 - 只有当平均强度极低且方差极小时才判定为纯黑
        if (stats['mean_intensity'] < self.thresholds['min_intensity'] and 
            stats['variance'] < self.thresholds['min_variance']):
            return True, "纯黑图像"
        
        # 检查过曝
        if stats['overexposed_ratio'] > self.thresholds['overexposure_ratio']:
            return True, "图像过曝"
        
        return False, ""
    
    def process_image(self, image: np.ndarray, is_overexposed: bool) -> np.ndarray:
        """处理图像
        
        Args:
            image: 原始图像数据
            is_overexposed: 是否过曝
            
        Returns:
            处理后的图像数据
        """
        if is_overexposed:
            # 对过曝图像进行对比度调整
            processed = np.zeros_like(image)
            for i in range(image.shape[0]):
                # 使用对比度受限的自适应直方图均衡
                processed[i] = exposure.equalize_adapthist(
                    image[i], 
                    clip_limit=0.03
                ) * 65535
            return processed.astype(np.uint16)
        else:
            # 标准化到16位范围
            normalized = ((image - image.min()) / (image.max() - image.min()) * 65535)
            return normalized.astype(np.uint16)
    
    def save_comparison(self, original: np.ndarray, processed: np.ndarray, 
                       filename: str, reason: str, output_dir: str):
        """保存处理前后的对比图
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            filename: 文件名
            reason: 异常原因
            output_dir: 输出目录
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'异常样本对比: {filename}\n原因: {reason}', fontsize=12, fontproperties=font)
        
        # 选择要显示的切片（开始、中间、结束）
        slices = [0, original.shape[0]//2, -1]
        
        for i, slice_idx in enumerate(slices):
            # 显示原始图像
            axes[0, i].imshow(original[slice_idx], cmap='gray')
            axes[0, i].set_title(f'原始图像 (切片 {slice_idx})', fontproperties=font)
            axes[0, i].axis('off')
            
            # 显示处理后的图像
            axes[1, i].imshow(processed[slice_idx], cmap='gray')
            axes[1, i].set_title(f'处理后图像 (切片 {slice_idx})', fontproperties=font)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        comparison_dir = os.path.join(output_dir, 'anomaly_comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        plt.savefig(os.path.join(comparison_dir, f'{os.path.splitext(filename)[0]}_comparison.png'))
        plt.close()
    
    def generate_report(self, output_dir: str):
        """生成统计报告
        
        Args:
            output_dir: 输出目录
        """
        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('数据集质量评估报告', fontsize=16, fontproperties=font)
        
        # 平均强度分布
        sns.histplot(data=self.stats_data['mean_intensities'], ax=axes[0, 0], bins=30)
        axes[0, 0].set_title('平均强度分布', fontproperties=font)
        axes[0, 0].set_xlabel('平均强度', fontproperties=font)
        axes[0, 0].set_ylabel('样本数量', fontproperties=font)
        
        # 方差分布
        sns.histplot(data=self.stats_data['variances'], ax=axes[0, 1], bins=30)
        axes[0, 1].set_title('方差分布', fontproperties=font)
        axes[0, 1].set_xlabel('方差', fontproperties=font)
        axes[0, 1].set_ylabel('样本数量', fontproperties=font)
        
        # 信噪比分布
        sns.histplot(data=self.stats_data['snrs'], ax=axes[1, 0], bins=30)
        axes[1, 0].set_title('信噪比分布', fontproperties=font)
        axes[1, 0].set_xlabel('信噪比', fontproperties=font)
        axes[1, 0].set_ylabel('样本数量', fontproperties=font)
        
        # 结构熵分布
        sns.histplot(data=self.stats_data['entropies'], ax=axes[1, 1], bins=30)
        axes[1, 1].set_title('结构熵分布', fontproperties=font)
        axes[1, 1].set_xlabel('结构熵', fontproperties=font)
        axes[1, 1].set_ylabel('样本数量', fontproperties=font)
        
        # 添加异常样本统计信息
        anomaly_text = f"总样本数: {len(self.stats_data['filenames'])}\n"
        anomaly_text += f"异常样本数: {len(self.stats_data['anomalies'])}\n"
        if self.stats_data['anomalies']:
            reasons = {}
            for _, reason in self.stats_data['anomalies']:
                reasons[reason] = reasons.get(reason, 0) + 1
            anomaly_text += "\n异常原因统计:\n"
            for reason, count in reasons.items():
                anomaly_text += f"{reason}: {count}个样本\n"
        
        plt.figtext(0.1, 0.02, anomaly_text, fontsize=10, fontproperties=font,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_report.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

def normalize_tif_batch(input_dir: str, output_dir: str, num_threads: int = 4) -> Dict:
    """批量处理TIF文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        num_threads: 线程数
        
    Returns:
        处理结果统计
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = TifAnalyzer()
    
    # 获取所有TIF文件
    tif_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.tif', '.tiff'))]
    
    if not tif_files:
        raise ValueError(f"在{input_dir}中未找到TIF文件")
    
    logging.info(f"找到{len(tif_files)}个TIF文件")
    
    # 处理结果统计
    results = {
        'total': len(tif_files),
        'successful': [],
        'failed': [],
        'anomalies': []
    }
    
    def process_file(filename: str) -> None:
        """处理单个文件"""
        try:
            # 读取图像
            input_path = os.path.join(input_dir, filename)
            image = tifffile.imread(input_path)
            
            # 计算统计指标
            stats = analyzer.calculate_stats(image)
            
            # 更新统计数据
            analyzer.stats_data['mean_intensities'].append(stats['mean_intensity'])
            analyzer.stats_data['variances'].append(stats['variance'])
            analyzer.stats_data['snrs'].append(stats['snr'])
            analyzer.stats_data['entropies'].append(stats['entropy'])
            analyzer.stats_data['filenames'].append(filename)
            
            # 检查是否异常
            is_anomaly, reason = analyzer.check_anomaly(image, stats, filename)
            
            if is_anomaly:
                analyzer.stats_data['anomalies'].append((filename, reason))
                results['anomalies'].append(f"{filename}: {reason}")
                
                if reason == "纯黑图像":
                    logging.warning(f"跳过纯黑图像: {filename}")
                    results['failed'].append(filename)
                    return
                
                # 处理异常图像
                processed = analyzer.process_image(
                    image, 
                    reason == "图像过曝"
                )
                
                # 保存对比图
                analyzer.save_comparison(
                    image, processed, filename, reason, output_dir
                )
                
                # 保存处理后的图像
                output_filename = f"{os.path.splitext(filename)[0]}_processed.tif"
                
            else:
                # 正常样本，标准化处理
                processed = analyzer.process_image(image, False)
                output_filename = filename
            
            # 保存处理后的图像
            output_path = os.path.join(output_dir, output_filename)
            tifffile.imwrite(output_path, processed)
            
            results['successful'].append(filename)
            
        except Exception as e:
            logging.error(f"处理{filename}时出错: {str(e)}")
            results['failed'].append(filename)
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(executor.map(process_file, tif_files))
    
    # 生成报告
    analyzer.generate_report(output_dir)
    
    # 输出处理结果
    logging.info(f"""
    处理完成:
    - 总样本数: {len(tif_files)}
    - 成功处理: {len(results['successful'])}
    - 异常样本: {len(results['anomalies'])}
    - 处理失败: {len(results['failed'])}
    """)
    
    return results

if __name__ == "__main__":
    # 示例用法
    input_dir = "F:/科研/光学成像/hylfm-net/光场仿真数据集/自拟数据集/ER"
    output_dir = "processed_data"
    
    try:
        results = normalize_tif_batch(input_dir, output_dir)
        print("处理完成！")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
