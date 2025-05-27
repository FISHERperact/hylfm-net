import sys
import os
import time
import subprocess
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QProgressDialog
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QIcon
from ui_main import Ui_MainWindow

# 导入功能模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 设置应用程序图标路径
LOGO_PATH = os.path.join(parent_dir, "logo.png")

try:
    from Crawler.url_downloader import Downloader
    from Crawler.url_downloader import download_file
    from Collect_data.collect_mrc_files import collect_microtubule_mrc_files
    from Format_conversion.mrc_to_tif_converter import mrc_to_tif, batch_convert_mrc_to_tif
    from Format_conversion.mat_to_tif_converter import mat_to_tif as convert_mat_to_tif
    from Format_conversion.tif_normalization import normalize_tif_batch  # 导入新的TIF标准化模块
    from Wfm_scan.process_wfm_convolution import process_mrc_files as process_wfm
    from Modify.rotate_tif_images_90 import rotate_tif_images_counterclockwise_90 as rotate_image
    from Modify.crop_mrc_files import crop_mrc_file as crop_image
    from Modify.split_tif_volumes import split_tif_volume as extract_layers
except ImportError as e:
    print(f"导入模块时出错: {e}")
    sys.exit(1)

class WorkerThread(QThread):
    """工作线程类，用于执行耗时操作"""
    finished = Signal(object)  # 完成信号，可传递结果数据
    error = Signal(str)  # 错误信号
    progress = Signal(str)  # 进度信号
    
    def __init__(self, task_func=None, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self._should_stop = False
        
    def set_task(self, task_func, *args, **kwargs):
        """设置任务函数和参数"""
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            if self.task_func:
                # 传递进度信号给任务函数，使其能发送进度更新
                if 'progress_signal' not in self.kwargs:
                    self.kwargs['progress_signal'] = self.progress.emit
                
                # 执行任务函数并获取结果
                self.result = self.task_func(*self.args, **self.kwargs)
                # 发送包含结果的完成信号
                self.finished.emit(self.result)
            else:
                self.error.emit("没有设置任务函数")
        except Exception as e:
            self.error.emit(str(e))
            
    def stop(self):
        """标记线程应该停止"""
        self._should_stop = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 设置窗口标题
        self.setWindowTitle("鱼天鸿毕业设计-图像处理工具")
        
        # 连接信号和槽
        self.setup_connections()
        
        # 当前工作线程
        self.current_thread = None
        
        # 当前下载线程
        self.download_thread = None
        
        # 下载速度计算参数
        self.last_update_time = time.time()
        self.last_downloaded = 0
        
    def setup_connections(self):
        """设置信号槽连接"""
        # 下载数据集页面
        self.ui.save_path_button.clicked.connect(
            lambda: self.select_directory(self.ui.save_path_input))
        self.ui.download_button.clicked.connect(self.start_download)
        
        # 提取MRC文件页面
        self.ui.zip_path_button.clicked.connect(
            lambda: self.select_directory(self.ui.zip_path_input))
        self.ui.extract_path_button.clicked.connect(
            lambda: self.select_directory(self.ui.extract_path_input))
        self.ui.extract_button.clicked.connect(self.extract_mrc)
        
        # 生成三维数据页面 - 修改为选择目录
        self.ui.input_3d_button.clicked.connect(
            lambda: self.select_directory(self.ui.input_3d_path))
        self.ui.output_3d_button.clicked.connect(
            lambda: self.select_directory(self.ui.output_3d_path))
        self.ui.convert_3d_button.clicked.connect(self.convert_to_3d)
        
        # MRC转TIF页面
        self.ui.mrc_input_button.clicked.connect(self.select_mrc_input)
        self.ui.mrc_output_button.clicked.connect(self.select_mrc_output)
        self.ui.mrc_convert_button.clicked.connect(self.process_mrc_to_tif)
        
        # MAT转TIF页面
        self.ui.mat_input_button.clicked.connect(
            lambda: self.select_file(self.ui.mat_input_path, "MAT文件 (*.mat)"))
        self.ui.mat_output_button.clicked.connect(
            lambda: self.select_directory(self.ui.mat_output_path))
        self.ui.mat_convert_button.clicked.connect(self.convert_mat_to_tif)
        
        # TIF标准化页面
        self.ui.tif_norm_input_button.clicked.connect(
            lambda: self.select_directory(self.ui.tif_norm_input_path))
        self.ui.tif_norm_output_button.clicked.connect(
            lambda: self.select_directory(self.ui.tif_norm_output_path))
        self.ui.tif_norm_button.clicked.connect(self.process_tif_normalization)
        
        # 宽场显微镜效果页面
        self.ui.wfm_input_button.clicked.connect(
            lambda: self.select_directory(self.ui.wfm_input_path))
        self.ui.wfm_output_button.clicked.connect(
            lambda: self.select_directory(self.ui.wfm_output_path))
        self.ui.wfm_process_button.clicked.connect(self.process_wfm_effect)
        
        # 图像修改页面
        self.ui.modify_input_button.clicked.connect(
            lambda: self.select_directory(self.ui.modify_input_path))
        self.ui.modify_output_button.clicked.connect(
            lambda: self.select_directory(self.ui.modify_output_path))
        self.ui.modify_process_button.clicked.connect(self.process_image_modification)
        
    def select_file(self, line_edit, file_filter):
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文件",
            "",
            file_filter
        )
        if file_path:
            line_edit.setText(file_path)
            
    def select_directory(self, line_edit):
        """选择目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择目录",
            ""
        )
        if dir_path:
            line_edit.setText(dir_path)
            
    def start_worker(self, task_func, finished_callback=None, error_callback=None, progress_callback=None):
        """启动工作线程
        
        参数:
            task_func: 要执行的任务函数
            finished_callback: 任务完成时的回调函数
            error_callback: 任务出错时的回调函数
            progress_callback: 任务进度更新时的回调函数
        
        返回:
            WorkerThread: 创建的工作线程对象
        """
        # 创建工作线程
        worker = WorkerThread(task_func)
        
        # 连接信号
        if finished_callback:
            worker.finished.connect(finished_callback)
        else:
            worker.finished.connect(lambda result: QMessageBox.information(self, "成功", "处理完成"))
        
        if error_callback:
            worker.error.connect(error_callback)
        else:
            worker.error.connect(lambda msg: QMessageBox.critical(self, "错误", f"处理出错: {msg}"))
        
        if progress_callback:
            worker.progress.connect(progress_callback)
        
        # 启动线程
        worker.start()
        
        return worker
        
    def update_download_progress(self, downloaded, total):
        """更新下载进度"""
        if total > 0:
            # 转换为MB
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            
            # 计算进度百分比
            progress = int((downloaded / total) * 100)
            self.ui.progress_bar.setValue(progress)
            
            # 设置格式，保持白色文本
            format_text = f"{progress}% - {downloaded_mb:.2f}/{total_mb:.2f} MB"
            self.ui.progress_bar.setFormat(format_text)
            
            # 计算下载速度
            now = time.time()
            elapsed = now - self.last_update_time
            if elapsed > 0:
                speed = (downloaded - self.last_downloaded) / elapsed
                speed_kb = speed / 1024
                
                if speed_kb < 1024:
                    speed_text = f"{speed_kb:.1f} KB/s"
                else:
                    speed_text = f"{speed_kb/1024:.1f} MB/s"
                    
                self.ui.status_label.setText(f"正在下载... 当前速度: {speed_text}")
                
                # 打印进度信息，方便调试
                print(f"下载进度: {progress}% - {downloaded_mb:.2f}/{total_mb:.2f} MB，速度: {speed_text}")
            
            # 保存当前值用于下次计算速度
            self.last_update_time = now
            self.last_downloaded = downloaded
            
            # 更新状态文本
            if progress == 0:
                self.ui.status_label.setText("正在连接服务器...")
            elif progress == 100:
                self.ui.status_label.setText("下载完成！")
        else:
            # 文件大小未知的情况
            self.ui.progress_bar.setValue(0)
            self.ui.progress_bar.setFormat("未知大小")
            self.ui.status_label.setText("正在下载...")
            
    def download_finished(self):
        """下载完成处理"""
        self.ui.progress_bar.setValue(100)
        self.ui.status_label.setText("下载完成！")
        self.ui.download_button.setEnabled(True)
        QMessageBox.information(self, "下载完成", "文件已成功下载！")
        
    def download_error(self, error_msg):
        """下载错误处理"""
        self.ui.status_label.setText(f"下载出错：{error_msg}")
        self.ui.download_button.setEnabled(True)
        QMessageBox.critical(self, "下载错误", f"下载过程中出现错误：\n{error_msg}")
            
    def start_download(self):
        """开始下载"""
        url = self.ui.url_input.text().strip()
        save_path = self.ui.save_path_input.text().strip()
        
        if not url or not save_path:
            self.ui.status_label.setText("错误：请输入下载链接和保存路径")
            return
            
        self.ui.download_button.setEnabled(False)
        self.ui.status_label.setText("正在连接服务器...")
        self.ui.progress_bar.setValue(0)
        
        try:
            # 确保路径是文件而不是目录
            if os.path.isdir(save_path):
                # 从URL中提取文件名
                filename = url.split('/')[-1]
                if '?' in filename:  # 处理带参数的URL
                    filename = filename.split('?')[0]
                    
                if not filename:
                    filename = "downloaded_file"
                
                save_path = os.path.join(save_path, filename)
            
            # 创建保存目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 初始化计时器
            self.last_update_time = time.time()
            self.last_downloaded = 0
            
            # 打印下载信息，方便调试
            print(f"开始下载 URL: {url}")
            print(f"保存到: {save_path}")
            
            # 创建并启动下载线程
            self.download_thread = Downloader(url, save_path)
            self.download_thread.progress_signal.connect(self.update_download_progress)
            self.download_thread.finished.connect(self.download_finished)
            self.download_thread.error.connect(self.download_error)
            self.download_thread.start()
            
            # 确认连接已经建立
            print("下载线程已启动，进度信号已连接")
            
        except Exception as e:
            self.ui.status_label.setText(f"下载出错：{str(e)}")
            self.ui.download_button.setEnabled(True)
            print(f"启动下载时出错: {str(e)}")
            
    def extract_mrc(self):
        """提取MRC文件"""
        source_path = self.ui.zip_path_input.text()
        target_path = self.ui.extract_path_input.text()
        
        if not source_path or not target_path:
            QMessageBox.warning(self, "警告", "请选择源文件夹和目标路径")
            return
            
        try:
            # 确保输出目录存在
            os.makedirs(target_path, exist_ok=True)
            
            # 查找所有MRC文件
            mrc_files = []
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith('.mrc'):
                        mrc_files.append(os.path.join(root, file))
            
            if not mrc_files:
                QMessageBox.warning(self, "警告", f"在 {source_path} 及其子目录中没有找到MRC文件")
                return
            
            # 创建进度对话框
            progress = QProgressDialog("正在提取MRC文件...", "取消", 0, len(mrc_files), self)
            progress.setWindowTitle("提取进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setMinimumDuration(0)
            
            # 记录成功和失败的文件
            successful_files = 0
            failed_files = []
            
            # 处理每个文件
            for i, mrc_file in enumerate(mrc_files):
                if progress.wasCanceled():
                    break
                
                # 更新进度对话框
                progress.setValue(i)
                progress.setLabelText(f"提取文件 {i+1}/{len(mrc_files)}: {os.path.basename(mrc_file)}")
                QApplication.processEvents()
                
                try:
                    # 按照序号创建新文件名
                    new_filename = f"sample({i+1}).mrc"
                    target_file = os.path.join(target_path, new_filename)
                    
                    # 如果目标文件已存在，添加数字后缀
                    if os.path.exists(target_file):
                        counter = 1
                        while os.path.exists(os.path.join(target_path, f"sample({i+1})_{counter}.mrc")):
                            counter += 1
                        target_file = os.path.join(target_path, f"sample({i+1})_{counter}.mrc")
                    
                    # 复制文件
                    import shutil
                    shutil.copy2(mrc_file, target_file)
                    successful_files += 1
                    print(f"成功提取: {mrc_file} -> {target_file}")
                
                except Exception as e:
                    print(f"提取文件失败: {mrc_file}, 错误: {str(e)}")
                    failed_files.append((os.path.basename(mrc_file), str(e)))
            
            # 完成处理
            progress.setValue(len(mrc_files))
            QApplication.processEvents()
            
            # 显示处理结果
            message = f"提取完成！\n\n"
            message += f"成功提取: {successful_files} / {len(mrc_files)} 个文件\n"
            
            if failed_files:
                message += f"\n失败的文件 ({len(failed_files)}):\n"
                for filename, error in failed_files[:10]:  # 只显示前10个失败文件
                    message += f"  - {filename}: {error}\n"
                if len(failed_files) > 10:
                    message += f"  ... 以及其他 {len(failed_files) - 10} 个文件\n"
            
            QMessageBox.information(self, "提取结果", message)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提取MRC文件时出错: {str(e)}")
        
    def convert_to_3d(self):
        """转换为3D数据"""
        input_path = self.ui.input_3d_path.text()
        output_path = self.ui.output_3d_path.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入目录和输出路径")
            return
        
        try:
            # 直接进行批处理，不再检查是否为目录
            self.batch_process_mrc_files(input_path, output_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成3D数据时出错: {str(e)}")
    
    def process_single_mrc_file(self, input_path, output_path):
        """处理单个MRC文件"""
        try:
            # 检查输入文件是否存在
            if not os.path.exists(input_path):
                QMessageBox.critical(self, "错误", f"输入文件不存在: {input_path}")
                return False
            
            # 检查输出目录是否存在，如果不存在则创建
            os.makedirs(output_path, exist_ok=True)
            
            # 获取文件名，不带扩展名
            input_file_name = os.path.basename(input_path)
            input_name_without_ext = os.path.splitext(input_file_name)[0]
            
            # 构建完整的输出文件路径
            output_file = os.path.join(output_path, f"{input_name_without_ext}_3d_21x483x483.mrc")
            
            # 构建命令
            python_exe = sys.executable  # 当前Python解释器路径
            script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "extract_lines_to_3d.py")
            
            # 检查脚本文件是否存在
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "错误", f"处理脚本不存在: {script_path}")
                return False
            
            cmd = [
                python_exe,
                script_path,
                input_path,
                "--output", output_file,
                "--extraction-method", "ridge",
                "--z-distribution", "wave",
                "--layers", "21",
                "--target-width", "483",
                "--target-height", "483",
                "--wave-frequency", "2.0",
                "--wave-amplitude", "0.8",
                "--tilt-factor", "0.3",
                "--z-thickness", "1",
                "--z-blur", "1.5",
                "--z-stretch", "1.0",
                "--preview"
            ]
            
            # 创建进度对话框
            progress = QProgressDialog("正在生成3D数据...", "取消", 0, 100, self)
            progress.setWindowTitle("处理进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setMinimumDuration(0)
            progress.setValue(10)  # 初始进度
            QApplication.processEvents()
            
            # 执行命令
            print("执行命令:", " ".join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 更新进度
            progress.setValue(30)
            QApplication.processEvents()
            
            # 等待命令执行完成
            stdout, stderr = process.communicate()
            
            # 检查命令执行结果
            if process.returncode != 0:
                error_msg = stderr if stderr else "未知错误"
                QMessageBox.critical(self, "错误", f"处理失败: {error_msg}")
                print(f"处理失败: {error_msg}")
                return False
            
            # 检查输出文件是否生成
            if not os.path.exists(output_file):
                QMessageBox.critical(self, "错误", "输出文件未生成")
                return False
            
            progress.setValue(100)
            QMessageBox.information(self, "成功", f"3D数据生成成功！\n输出文件: {output_file}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理过程中出现错误: {str(e)}")
            print(f"处理过程中出现错误: {str(e)}")
            return False
    
    def batch_process_mrc_files(self, input_dir, output_dir):
        """批量处理目录中的所有MRC文件"""
        try:
            # 查找输入目录中的所有.mrc文件
            mrc_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.mrc') and "_3d_" not in file:  # 排除已经生成的3D文件
                        mrc_files.append(os.path.join(root, file))
            
            if not mrc_files:
                QMessageBox.warning(self, "警告", f"在 {input_dir} 中没有找到MRC文件")
                return
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建进度对话框
            progress = QProgressDialog("正在批量处理MRC文件...", "取消", 0, len(mrc_files), self)
            progress.setWindowTitle("批处理进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setMinimumDuration(0)
            
            # 记录成功和失败的文件
            successful_files = []
            failed_files = []
            
            # 处理每个文件
            for i, mrc_file in enumerate(mrc_files):
                if progress.wasCanceled():
                    break
                
                # 更新进度对话框
                progress.setValue(i)
                progress.setLabelText(f"处理文件 {i+1}/{len(mrc_files)}: {os.path.basename(mrc_file)}")
                QApplication.processEvents()
                
                try:
                    # 保留原始目录结构
                    rel_path = os.path.relpath(os.path.dirname(mrc_file), input_dir)
                    if rel_path == '.':  # 如果是根目录
                        target_dir = output_dir
                    else:
                        target_dir = os.path.join(output_dir, rel_path)
                    
                    # 确保目标目录存在
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # 获取文件名，不带扩展名
                    input_file_name = os.path.basename(mrc_file)
                    input_name_without_ext = os.path.splitext(input_file_name)[0]
                    
                    # 构建完整的输出文件路径
                    output_file = os.path.join(target_dir, f"{input_name_without_ext}_3d_21x483x483.mrc")
                    
                    # 如果输出文件已存在，询问是否覆盖
                    if os.path.exists(output_file):
                        # 跳过已存在的文件
                        print(f"文件已存在，跳过: {output_file}")
                        continue
                    
                    # 构建命令
                    python_exe = sys.executable  # 当前Python解释器路径
                    script_path = os.path.join(parent_dir, "Elevating_dimension", "extract_lines_to_3d.py")
                    
                    cmd = [
                        python_exe,
                        script_path,
                        mrc_file,
                        "--output", output_file,
                        "--extraction-method", "ridge",
                        "--z-distribution", "wave",
                        "--layers", "21",
                        "--target-width", "483",
                        "--target-height", "483",
                        "--wave-frequency", "2.0",
                        "--wave-amplitude", "0.8",
                        "--tilt-factor", "0.3",
                        "--z-thickness", "1",
                        "--z-blur", "1.5",
                        "--z-stretch", "1.0",
                        "--preview"
                    ]
                    
                    # 执行命令
                    print(f"处理文件 {i+1}/{len(mrc_files)}: {os.path.basename(mrc_file)}")
                    print("执行命令:", " ".join(cmd))
                    
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    stdout, stderr = process.communicate()
                    
                    # 检查命令执行结果
                    if process.returncode != 0:
                        print(f"处理文件失败: {os.path.basename(mrc_file)}, 错误: {stderr}")
                        failed_files.append(f"{os.path.basename(mrc_file)} (处理失败)")
                    else:
                        successful_files.append(os.path.basename(mrc_file))
                        print(f"成功处理文件: {os.path.basename(mrc_file)}")
                
                except Exception as e:
                    print(f"处理文件出错: {os.path.basename(mrc_file)}, 错误: {str(e)}")
                    failed_files.append(f"{os.path.basename(mrc_file)} ({str(e)})")
            
            # 完成处理
            progress.setValue(len(mrc_files))
            QApplication.processEvents()
            
            # 显示处理结果
            message = f"批处理完成！\n\n"
            message += f"成功处理: {len(successful_files)} / {len(mrc_files)} 个文件\n\n"
            
            if failed_files:
                message += f"失败的文件 ({len(failed_files)}):\n"
                for file in failed_files[:10]:  # 只显示前10个失败文件
                    message += f"  - {file}\n"
                if len(failed_files) > 10:
                    message += f"  ... 以及其他 {len(failed_files) - 10} 个文件\n"
            
            QMessageBox.information(self, "批处理结果", message)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量处理MRC文件时出错: {str(e)}")
        
    def process_mrc_to_tif(self):
        """
        处理MRC到TIF的转换
        """
        # 获取输入和输出目录
        input_dir = self.ui.lineEdit_mrc_input.text().strip()
        output_dir = self.ui.lineEdit_mrc_output.text().strip()
        
        # 检查输入路径
        if not input_dir:
            self.message_box("错误", "请选择输入MRC文件或目录")
            return
        
        # 检查输出路径
        if not output_dir:
            self.message_box("错误", "请选择输出目录")
            return
        
        # 检查输入是文件还是目录
        input_is_dir = os.path.isdir(input_dir)
        
        try:
            if input_is_dir:
                # 批量处理目录
                self.ui.label_status.setText("正在批量转换MRC文件...")
                QApplication.processEvents()
                
                from Format_conversion.mrc_to_tif_converter import batch_convert_mrc_to_tif
                successful, failed = batch_convert_mrc_to_tif(input_dir, output_dir)
                
                # 构建结果消息
                message = f"批量转换完成!\n"
                message += f"成功转换: {len(successful)} 个文件\n"
                message += f"失败转换: {len(failed)} 个文件\n"
                
                if failed:
                    message += "\n失败的文件:\n"
                    for i, (file, error) in enumerate(failed[:5], 1):  # 只显示前5个失败
                        message += f"{i}. {os.path.basename(file)}: {error}\n"
                    
                    if len(failed) > 5:
                        message += f"...及其他 {len(failed) - 5} 个文件\n"
                
                message += f"\n输出目录: {output_dir}"
                
            else:
                # 处理单个文件
                self.ui.label_status.setText(f"正在转换MRC文件: {os.path.basename(input_dir)}...")
                QApplication.processEvents()
                
                from Format_conversion.mrc_to_tif_converter import mrc_to_tif
                
                # 如果输出是目录，则创建输出文件名
                if os.path.isdir(output_dir):
                    filename = os.path.basename(input_dir)
                    name_without_ext = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_dir, f"{name_without_ext}.tif")
                else:
                    output_file = output_dir
                
                # 转换文件
                mrc_to_tif(input_dir, output_file)
                
                message = f"转换成功!\n"
                message += f"输入: {input_dir}\n"
                message += f"输出: {output_file}"
            
            # 显示成功消息
            self.ui.label_status.setText("转换完成")
            self.message_box("成功", message)
            
        except Exception as e:
            # 显示错误消息
            self.ui.label_status.setText("转换失败")
            error_message = f"处理过程中发生错误: {str(e)}"
            self.message_box("错误", error_message)
            logging.error(f"MRC到TIF转换失败: {str(e)}", exc_info=True)

    def select_mrc_input(self):
        """
        选择MRC输入文件或目录
        """
        options = QFileDialog.Options()
        
        # 根据单个文件/批处理模式决定对话框类型
        if self.ui.checkBox_mrc_batch.isChecked():
            # 批处理模式 - 选择目录
            input_path = QFileDialog.getExistingDirectory(
                self, "选择包含MRC文件的目录", "", options=options
            )
        else:
            # 单文件模式 - 选择文件
            input_path, _ = QFileDialog.getOpenFileName(
                self, "选择MRC文件", "", "MRC文件 (*.mrc);;所有文件 (*)", options=options
            )
        
        if input_path:
            self.ui.lineEdit_mrc_input.setText(input_path)

    def select_mrc_output(self):
        """
        选择MRC转换的输出目录或文件
        """
        options = QFileDialog.Options()
        
        if self.ui.checkBox_mrc_batch.isChecked() or os.path.isdir(self.ui.lineEdit_mrc_input.text()):
            # 如果是批处理模式或输入是目录，则输出必须是目录
            output_dir = QFileDialog.getExistingDirectory(
                self, "选择输出目录", "", options=options
            )
            if output_dir:
                self.ui.lineEdit_mrc_output.setText(output_dir)
        else:
            # 单文件模式 - 可以选择保存文件
            input_file = self.ui.lineEdit_mrc_input.text()
            suggested_name = ""
            
            if input_file:
                filename = os.path.basename(input_file)
                name_without_ext = os.path.splitext(filename)[0]
                suggested_name = f"{name_without_ext}.tif"
            
            output_file, _ = QFileDialog.getSaveFileName(
                self, "保存为TIF文件", suggested_name, "TIF文件 (*.tif);;所有文件 (*)", options=options
            )
            
            if output_file:
                self.ui.lineEdit_mrc_output.setText(output_file)

    def convert_mat_to_tif(self):
        """MAT转TIF"""
        input_path = self.ui.mat_input_path.text()
        output_path = self.ui.mat_output_path.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入文件和输出路径")
            return
            
        self.start_worker(convert_mat_to_tif, input_path, output_path)
        
    def process_wfm_effect(self):
        """处理宽场显微镜效果"""
        input_path = self.ui.wfm_input_path.text()
        output_path = self.ui.wfm_output_path.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入路径和输出路径")
            return
        
        # 检查输入路径是否为目录
        if not os.path.isdir(input_path):
            QMessageBox.warning(self, "警告", "输入路径必须是一个文件夹")
            return
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 获取PSF文件路径
        psf_path = os.path.join(parent_dir, "Wfm_scan", "psf_wfm.tif")
        if not os.path.exists(psf_path):
            QMessageBox.warning(self, "警告", f"找不到PSF文件: {psf_path}")
            return
        
        try:
            # 递归查找所有TIF文件
            tif_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith((".tif", ".tiff")):
                        tif_files.append(os.path.join(root, file))
            
            if not tif_files:
                QMessageBox.warning(self, "警告", f"在 {input_path} 中没有找到TIF文件")
                return
            
            # 创建进度对话框
            progress = QProgressDialog("正在处理宽场效果...", "取消", 0, len(tif_files), self)
            progress.setWindowTitle("处理进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(True)
            progress.setMinimumDuration(0)
            
            # 记录成功和失败的文件
            successful_files = 0
            failed_files = []
            
            # 读取PSF
            import mrcfile
            from pathlib import Path
            from scipy.signal import convolve2d
            from tifffile import imread, imwrite
            import numpy as np
            
            # 读取PSF函数
            def read_psf(psf_path):
                try:
                    psf = imread(psf_path)
                    psf = psf.astype(np.float64)
                    # 归一化PSF，使其最大值为1
                    psf = psf / np.max(psf)
                    return psf
                except Exception as e:
                    print(f"读取PSF文件 {psf_path} 时出错: {str(e)}")
                    return None
            
            # 读取TIF函数
            def read_tif_file(tif_path):
                try:
                    img = imread(tif_path)
                    return img.astype(np.float64)
                except Exception as e:
                    print(f"读取TIF文件 {tif_path} 时出错: {str(e)}")
                    return None
            
            # 应用宽场卷积函数
            def apply_wfm_convolution(image_data, psf):
                # 检查图像是否为3D（图像堆栈）
                if len(image_data.shape) == 3:
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
                else:
                    # 单个2D图像
                    convolved_image = convolve2d(image_data, psf, mode='same')
                    # 归一化
                    if np.max(convolved_image) > 0:
                        convolved_image = convolved_image / np.max(convolved_image)
                    wfm_data = convolved_image
                
                return wfm_data
            
            # 读取PSF
            psf = read_psf(psf_path)
            if psf is None:
                QMessageBox.critical(self, "错误", f"无法读取PSF文件: {psf_path}")
                return
            
            # 处理每个文件
            for i, tif_file in enumerate(tif_files):
                if progress.wasCanceled():
                    break
                
                # 更新进度对话框
                progress.setValue(i)
                progress.setLabelText(f"处理文件 {i+1}/{len(tif_files)}: {os.path.basename(tif_file)}")
                QApplication.processEvents()
                
                try:
                    # 保留原始目录结构
                    rel_path = os.path.relpath(os.path.dirname(tif_file), input_path)
                    if rel_path == '.':  # 如果是根目录
                        target_dir = output_path
                    else:
                        target_dir = os.path.join(output_path, rel_path)
                    
                    # 确保目标目录存在
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # 获取文件名，不带扩展名
                    filename = os.path.basename(tif_file)
                    name_without_ext = os.path.splitext(filename)[0]
                    
                    # 创建输出文件路径
                    output_file = os.path.join(target_dir, f"{name_without_ext}_wfm.tif")
                    
                    # 读取TIF数据
                    img_data = read_tif_file(tif_file)
                    if img_data is None:
                        raise Exception(f"无法读取文件: {tif_file}")
                    
                    # 应用宽场卷积
                    wfm_data = apply_wfm_convolution(img_data, psf)
                    
                    # 保存处理后的图像
                    wfm_data = (wfm_data * 65535).astype(np.uint16)  # 转换为16位整数
                    imwrite(output_file, wfm_data)
                    
                    successful_files += 1
                    print(f"成功处理: {tif_file} -> {output_file}")
                    
                except Exception as e:
                    print(f"处理文件失败: {tif_file}, 错误: {str(e)}")
                    failed_files.append((os.path.basename(tif_file), str(e)))
            
            # 完成处理
            progress.setValue(len(tif_files))
            QApplication.processEvents()
            
            # 显示处理结果
            message = f"处理完成！\n\n"
            message += f"成功处理: {successful_files} / {len(tif_files)} 个文件\n"
            
            if failed_files:
                message += f"\n失败的文件 ({len(failed_files)}):\n"
                for filename, error in failed_files[:10]:  # 只显示前10个失败文件
                    message += f"  - {filename}: {error}\n"
                if len(failed_files) > 10:
                    message += f"  ... 以及其他 {len(failed_files) - 10} 个文件\n"
            
            QMessageBox.information(self, "处理结果", message)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理宽场效果时出错: {str(e)}")
        
    def process_image_modification(self):
        """处理图像修改"""
        input_path = self.ui.modify_input_path.text()
        output_path = self.ui.modify_output_path.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入路径和输出路径")
            return
            
        # 获取参数
        angle = self.ui.angle_spin.value()
        crop_size = self.ui.crop_spin.value()
        depth = self.ui.depth_spin.value()
        
        # 检查输入路径是否为目录
        if not os.path.isdir(input_path):
            QMessageBox.warning(self, "警告", "输入路径必须是一个文件夹")
            return
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 查找所有 TIF 文件
        tif_files = []
        for file in os.listdir(input_path):
            if file.lower().endswith((".tif", ".tiff")):
                tif_files.append(os.path.join(input_path, file))
        
        if not tif_files:
            QMessageBox.warning(self, "警告", f"在 {input_path} 中没有找到 TIF 文件")
            return
        
        # 创建进度对话框
        progress = QProgressDialog("正在处理图像文件...", "取消", 0, len(tif_files), self)
        progress.setWindowTitle("处理进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setMinimumDuration(0)
        
        # 记录成功和失败的文件
        successful_files = 0
        failed_files = []
        
        # 处理每个文件
        for i, tif_file in enumerate(tif_files):
            if progress.wasCanceled():
                break
            
            # 更新进度对话框
            progress.setValue(i)
            progress.setLabelText(f"处理文件 {i+1}/{len(tif_files)}: {os.path.basename(tif_file)}")
            QApplication.processEvents()
            
            try:
                # 获取文件名，不带扩展名
                filename = os.path.basename(tif_file)
                name_without_ext = os.path.splitext(filename)[0]
                
                # 创建输出文件路径
                output_file = os.path.join(output_path, filename)
                
                # 根据参数执行相应操作
                if angle != 0:
                    # 修复函数调用，传递角度参数
                    rotate_image(tif_file, output_file, angle)
                elif crop_size > 0:
                    crop_image(tif_file, output_file, crop_size)
                elif depth > 1:
                    extract_layers(tif_file, output_file, depth)
                else:
                    # 如果没有设置任何参数，跳过此文件
                    continue
                
                successful_files += 1
                
            except Exception as e:
                print(f"处理文件失败: {tif_file}, 错误: {str(e)}")
                failed_files.append((os.path.basename(tif_file), str(e)))
        
        # 完成处理
        progress.setValue(len(tif_files))
        QApplication.processEvents()
        
        # 显示处理结果
        message = f"处理完成！\n\n"
        message += f"成功处理: {successful_files} / {len(tif_files)} 个文件\n"
        
        if failed_files:
            message += f"\n失败的文件 ({len(failed_files)}):\n"
            for filename, error in failed_files[:10]:  # 只显示前10个失败文件
                message += f"  - {filename}: {error}\n"
            if len(failed_files) > 10:
                message += f"  ... 以及其他 {len(failed_files) - 10} 个文件\n"
        
        QMessageBox.information(self, "处理结果", message)

    def process_tif_normalization(self):
        """处理三维TIF标准化"""
        input_path = self.ui.tif_norm_input_path.text()
        output_path = self.ui.tif_norm_output_path.text()
        
        if not input_path or not output_path:
            QMessageBox.warning(self, "警告", "请选择输入目录和输出目录")
            return
        
        # 检查输入路径是否为目录
        if not os.path.isdir(input_path):
            QMessageBox.warning(self, "警告", "输入路径必须是一个目录")
            return
        
        # 创建进度对话框
        self.progress_dialog = QProgressDialog("正在准备标准化TIF文件...", "取消", 0, 100, self)
        self.progress_dialog.setWindowTitle("标准化进度")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setValue(0)
        self.progress_dialog.canceled.connect(self.cancel_tif_normalization)
        
        # 设置线程数
        threads = min(os.cpu_count() or 4, 8)  # 最多使用8个线程
        
        # 定义标准化任务函数
        def tif_normalization_task(progress_signal=None):
            try:
                # 导入函数和模块
                from Format_conversion.tif_normalization import normalize_tif_batch
                import logging
                
                # 确保有进度信号
                if progress_signal is None:
                    progress_signal = lambda msg: None
                
                # 创建信号发送器类
                class SignalHandler(logging.Handler):
                    def __init__(self, signal_func):
                        super().__init__()
                        self.signal_func = signal_func
                        
                    def emit(self, record):
                        msg = self.format(record)
                        self.signal_func(msg)
                
                # 获取logger并添加信号处理器
                logger = logging.getLogger("TIF标准化")
                handler = SignalHandler(progress_signal)
                handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(handler)
                
                # 向进度显示发送初始消息
                progress_signal("开始准备文件...")
                
                # 开始处理
                results = normalize_tif_batch(input_path, output_path, threads)
                
                # 移除处理器，避免重复
                logger.removeHandler(handler)
                
                return results
            except Exception as e:
                raise Exception(f"TIF标准化处理失败: {str(e)}")
        
        # 定义进度更新函数
        def update_progress(message):
            if "处理文件" in message or "找到" in message or "开始处理" in message or "准备文件" in message:
                # 更新进度条状态文本
                self.progress_dialog.setLabelText(f"正在标准化: {message}")
            
        # 处理完成时的回调
        def normalization_finished(results):
            self.progress_dialog.setValue(100)
            
            # 显示处理结果
            message = f"标准化处理完成！\n\n"
            message += f"总共处理: {results['processed']} 个文件\n"
            message += f"成功处理: {len(results['successful'])} 个文件\n"
            message += f"异常样本: {len(results['outliers'])} 个文件（已处理但标记为异常）\n"
            message += f"处理失败: {len(results['failed']) - len(results['outliers'])} 个文件\n"
            
            # 处理失败的文件（不包括异常样本）
            real_failures = [msg for msg in results['failed'] if "异常样本" not in msg]
            if real_failures:
                message += f"\n处理失败的文件:\n"
                for i, msg in enumerate(real_failures[:5], 1):
                    message += f"  {i}. {msg}\n"
                if len(real_failures) > 5:
                    message += f"  ... 以及其他 {len(real_failures) - 5} 个文件\n"
            
            # 异常样本
            if results['outliers']:
                message += f"\n异常样本（已处理但质量异常）:\n"
                for i, (filename, reason) in enumerate(results['outliers'][:5], 1):
                    message += f"  {i}. {filename}: {reason}\n"
                if len(results['outliers']) > 5:
                    message += f"  ... 以及其他 {len(results['outliers']) - 5} 个样本\n"
            
            # 提示用户查看标准化结果
            message += f"\n所有处理完成的文件都已保存到输出目录:\n{output_path}\n"
            message += f"异常样本文件名中添加了\"_异常\"标记。"
            
            QMessageBox.information(self, "标准化结果", message)
            
            # 询问是否查看质量评估报告
            report_path = os.path.join(output_path, "quality_metrics_report.png")
            if os.path.exists(report_path):
                reply = QMessageBox.question(self, "查看报告", 
                                            "是否打开质量评估报告？",
                                            QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    # 使用系统默认程序打开报告
                    import webbrowser
                    webbrowser.open(os.path.abspath(report_path))
        
        # 处理错误时的回调
        def normalization_error(error_msg):
            QMessageBox.critical(self, "错误", error_msg)
        
        # 创建并启动工作线程
        self.tif_worker = WorkerThread()
        self.tif_worker.finished.connect(normalization_finished)
        self.tif_worker.error.connect(normalization_error)
        self.tif_worker.progress.connect(update_progress)
        self.tif_worker.set_task(tif_normalization_task)
        self.tif_worker.start()

    def cancel_tif_normalization(self):
        """取消TIF标准化处理"""
        if hasattr(self, 'tif_worker') and self.tif_worker.isRunning():
            # 标记线程应该停止
            self.tif_worker.stop()
            
            # 安全终止线程（不推荐使用terminate，除非必要）
            self.tif_worker.wait(1000)  # 等待1秒
            if self.tif_worker.isRunning():
                self.tif_worker.terminate()
                self.tif_worker.wait()
            
            QMessageBox.information(self, "已取消", "TIF标准化处理已取消")

def main():
    # 禁用Qt的样式表警告
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.stylesheet=false"
    
    app = QApplication(sys.argv)
    
    # 设置应用程序图标
    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QIcon(LOGO_PATH))
    else:
        print(f"警告: 找不到图标文件 {LOGO_PATH}")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 