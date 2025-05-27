from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QTabWidget, QSpinBox,
                               QFrame, QSizePolicy, QProgressBar, QScrollArea)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QPalette, QColor, QIcon, QDesktopServices, QPainter, QBrush

class Ui_MainWindow:
    def setupUi(self, MainWindow):
        # 设置主窗口
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)  # 增加窗口大小
        MainWindow.setMinimumSize(1000, 800)  # 设置最小尺寸
        
        # 设置主窗口样式
        MainWindow.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d2d;
            }
            QWidget {
                background-color: #2d2d2d;
            }
            QTabWidget::pane {
                border: none;
                background-color: #2d2d2d;
                border-radius: 0px;
            }
            QTabBar::tab {
                background-color: #252525;
                color: #ffffff;
                padding: 10px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
                padding: 5px;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 8px;
                min-height: 30px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 12px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                min-height: 30px;
                min-width: 100px;
                font-size: 12px;
            }
            QFrame {
                background-color: transparent;
                border-radius: 10px;
                padding: 10px;
            }
            QFrame#website-frame {
                background-color: #252525;
                border-radius: 5px;
                padding: 10px;
                margin: 5px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QWidget#centralwidget {
                background-color: #2d2d2d;
            }
        """)
        
        # 添加QProgressBar样式
        MainWindow.setStyleSheet(MainWindow.styleSheet() + """
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #3d3d3d;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
            QPushButton.link-button {
                background-color: transparent;
                color: #0078d4;
                text-align: left;
                padding: 5px;
                border: none;
                font-size: 12px;
                min-height: 20px;
            }
            QPushButton.link-button:hover {
                color: #1e88e5;
                text-decoration: underline;
            }
        """)
        
        # 创建中央部件
        self.centralwidget = QWidget(MainWindow)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)  # 减小边距
        self.main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("毕业设计-光场显微镜图像处理工具")
        title_label.setStyleSheet("""
            font-size: 24px;
            color: #ffffff;
            padding: 20px;
            font-weight: bold;
            background-color: #0078d4;
            border-radius: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)
        
        # 创建标签页
        self.tabWidget = QTabWidget()
        self.main_layout.addWidget(self.tabWidget)
        
        # 添加各个功能页面
        self.setup_download_tab()
        self.setup_extract_tab()
        self.setup_3d_conversion_tab()
        self.setup_mrc_tab()
        self.setup_mat_tab()
        self.setup_tif_normalization_tab()
        self.setup_wfm_tab()
        self.setup_modify_tab()
        
        MainWindow.setCentralWidget(self.centralwidget)
        
    def create_file_input_group(self, label_text, show_browse_button=True):
        """创建文件输入组"""
        group = QFrame()
        group.setStyleSheet("QFrame { background-color: transparent; padding: 0px; }")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        label = QLabel(label_text)
        label.setMinimumWidth(100)
        input_field = QLineEdit()
        input_field.setMinimumWidth(500)  # 增加输入框宽度
        
        if show_browse_button:
            browse_button = QPushButton("浏览")
            browse_button.setMinimumWidth(100)
            layout.addWidget(label)
            layout.addWidget(input_field)
            layout.addWidget(browse_button)
            return group, input_field, browse_button
        else:
            # 添加微透镜阵列图标
            class MicrolensIcon(QLabel):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setFixedSize(32, 32)
                    self.setStyleSheet("""
                        QLabel {
                            border: 2px solid #0078d4;
                            border-radius: 16px;
                            background-color: #252525;
                        }
                    """)
                
                def paintEvent(self, event):
                    super().paintEvent(event)
                    painter = QPainter(self)
                    painter.setRenderHint(QPainter.Antialiasing)
                    
                    # 定义点的颜色和大小
                    dot_color = QColor("#0078d4")
                    dot_size = 4
                    spacing = 8
                    
                    # 绘制点阵
                    for x in range(spacing//2, self.width(), spacing):
                        for y in range(spacing//2, self.height(), spacing):
                            painter.setBrush(QBrush(dot_color))
                            painter.setPen(Qt.NoPen)
                            painter.drawEllipse(x - dot_size//2, y - dot_size//2, dot_size, dot_size)
            
            icon = MicrolensIcon()
            layout.addWidget(label)
            layout.addWidget(input_field)
            layout.addWidget(icon)
            return group, input_field, None
        
    def setup_download_tab(self):
        """设置下载数据集页面"""
        tab = QWidget()
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        # 创建内容容器
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(20)
        
        # 常用数据集网站区域
        websites_frame = QFrame()
        websites_frame.setObjectName("website-frame")
        websites_layout = QVBoxLayout(websites_frame)
        websites_layout.setSpacing(15)
        websites_layout.setContentsMargins(15, 20, 15, 20)
        
        # 添加标题
        websites_title = QLabel("常用数据集网站")
        websites_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
            background-color: #1e88e5;
            border-radius: 5px;
        """)
        websites_layout.addWidget(websites_title)
        
        # 添加网站链接
        websites = [
            ("Figshare - 科研数据共享平台", "https://figshare.com"),
            ("BioStudies - 生物研究数据库", "https://www.ebi.ac.uk/biostudies/"),
            ("Zenodo - 开放科学数据平台", "https://zenodo.org"),
            ("Cell Image Library - 细胞图像库", "http://www.cellimagelibrary.org"),
            ("IDR - Image Data Resource", "https://idr.openmicroscopy.org"),
            ("NeuroMorpho - 神经形态数据库", "http://neuromorpho.org")
        ]
        
        for name, url in websites:
            link_button = QPushButton(name)
            link_button.setProperty("url", url)
            link_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #0078d4;
                    text-align: left;
                    padding: 8px;
                    border: none;
                    font-size: 13px;
                }
                QPushButton:hover {
                    color: #1e88e5;
                    text-decoration: underline;
                    background-color: rgba(30, 136, 229, 0.1);
                    border-radius: 5px;
                }
            """)
            link_button.clicked.connect(lambda checked=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            websites_layout.addWidget(link_button)
        
        layout.addWidget(websites_frame)
        
        # 下载控件区域
        download_frame = QFrame()
        download_frame.setObjectName("website-frame")
        download_layout = QVBoxLayout(download_frame)
        download_layout.setSpacing(15)
        
        # URL输入组
        url_group, self.url_input, _ = self.create_file_input_group("下载链接：", show_browse_button=False)
        download_layout.addWidget(url_group)
        
        # 保存路径组
        save_group, self.save_path_input, self.save_path_button = self.create_file_input_group("保存路径：")
        download_layout.addWidget(save_group)
        
        # 进度显示区域
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m MB")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #3d3d3d;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # 添加状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("""
            color: #ffffff;
            font-size: 13px;
            font-weight: bold;
            padding: 5px;
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        """)
        progress_layout.addWidget(self.status_label)
        
        download_layout.addWidget(progress_frame)
        
        # 下载按钮
        self.download_button = QPushButton("开始下载")
        self.download_button.setMinimumHeight(40)
        download_layout.addWidget(self.download_button)
        
        layout.addWidget(download_frame)
        layout.addStretch()
        
        # 设置滚动区域的内容
        scroll.setWidget(content_widget)
        
        # 创建主布局并添加滚动区域
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
        self.tabWidget.addTab(tab, "数据集下载")
        
    def setup_extract_tab(self):
        """设置提取MRC文件页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 源路径组
        zip_group, self.zip_path_input, self.zip_path_button = self.create_file_input_group("源文件夹路径：")
        layout.addWidget(zip_group)
        
        # 提取路径组
        extract_group, self.extract_path_input, self.extract_path_button = self.create_file_input_group("提取目标路径：")
        layout.addWidget(extract_group)
        
        # 提取按钮
        self.extract_button = QPushButton("开始提取并重命名")
        self.extract_button.setMinimumHeight(40)
        layout.addWidget(self.extract_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可从指定目录及其所有子目录中提取MRC文件，并按顺序重命名。\n"
            "- 自动遍历所有子文件夹，查找所有MRC文件\n"
            "- 提取到目标路径并按顺序重命名为: sample(1).mrc, sample(2).mrc, ...\n"
            "- 原始目录结构不会保留，所有文件将存放在同一目录下\n"
            "- 适用于从已解压的数据集文件夹中批量提取MRC文件"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "提取MRC文件")
        
    def setup_3d_conversion_tab(self):
        """设置生成三维数据页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 输入文件组
        input_group, self.input_3d_path, self.input_3d_button = self.create_file_input_group("MRC目录路径：")
        layout.addWidget(input_group)
        
        # 输出路径组
        output_group, self.output_3d_path, self.output_3d_button = self.create_file_input_group("输出目录路径：")
        layout.addWidget(output_group)
        
        # 转换按钮
        self.convert_3d_button = QPushButton("批量生成3D数据")
        self.convert_3d_button.setMinimumHeight(40)
        layout.addWidget(self.convert_3d_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可批量处理指定目录中的所有MRC文件，将它们转换为3D数据。\n"
            "- 输入目录可以包含多个MRC文件\n"
            "- 处理后会保留原始的目录结构\n"
            "- 默认生成21x483x483大小的3D数据"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        self.tabWidget.addTab(tab, "生成三维数据")
        
    def setup_mrc_tab(self):
        """设置MRC转TIF页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        # 输入路径组 - 改为目录选择
        input_group, self.mrc_input_path, self.mrc_input_button = self.create_file_input_group("MRC文件目录：")
        layout.addWidget(input_group)
        
        # 输出路径组
        output_group, self.mrc_output_path, self.mrc_output_button = self.create_file_input_group("输出目录路径：")
        layout.addWidget(output_group)
        
        # 转换按钮
        self.mrc_convert_button = QPushButton("批量转换为TIF")
        self.mrc_convert_button.setMinimumHeight(40)
        layout.addWidget(self.mrc_convert_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可批量处理指定目录及其子目录中的所有MRC文件，将它们转换为TIF格式。\n"
            "- 自动遍历所有子文件夹，查找所有MRC文件\n"
            "- 转换后会保留原始的目录结构\n"
            "- 转换后的文件名格式为: 原文件名.tif"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "MRC转TIF")
        
    def setup_mat_tab(self):
        """设置MAT转TIF页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        # 输入路径组
        input_group, self.mat_input_path, self.mat_input_button = self.create_file_input_group("MAT文件：")
        layout.addWidget(input_group)
        
        # 输出路径组
        output_group, self.mat_output_path, self.mat_output_button = self.create_file_input_group("输出路径：")
        layout.addWidget(output_group)
        
        # 转换按钮
        self.mat_convert_button = QPushButton("转换为TIF")
        self.mat_convert_button.setMinimumHeight(40)
        layout.addWidget(self.mat_convert_button)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "MAT转TIF")
        
    def setup_tif_normalization_tab(self):
        """设置三维TIF标准化页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 输入目录组
        input_group, self.tif_norm_input_path, self.tif_norm_input_button = self.create_file_input_group("三维TIF文件目录：")
        layout.addWidget(input_group)
        
        # 输出目录组
        output_group, self.tif_norm_output_path, self.tif_norm_output_button = self.create_file_input_group("标准化输出目录：")
        layout.addWidget(output_group)
        
        # 标准化按钮
        self.tif_norm_button = QPushButton("开始标准化处理")
        self.tif_norm_button.setMinimumHeight(40)
        layout.addWidget(self.tif_norm_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可对三维TIF数据进行标准化处理和质量评估。\n\n"
            "标准化处理步骤:\n"
            "1. 对图像进行局部限幅，裁剪异常值，防止归一化过程中出现梯度塌陷或背景过亮的伪信号\n"
            "2. 将图像归一化到[0,1]范围内: I' = (I - Imin) / (Imax - Imin)\n"
            "3. 生成原始与标准化后图像的直方图对比\n\n"
            "质量评估指标:\n"
            "- 结构熵: 评估图像的复杂度和信息量\n"
            "- 对比度: 检测图像的明暗差异\n"
            "- 信噪比: 衡量图像信号与噪声比例\n"
            "- 方差: 评估图像像素分布的离散程度\n\n"
            "处理结束后将生成质量评估报告，显示所有样本的质量分布情况，并自动剔除异常样本。"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "三维TIF标准化")
        
    def setup_wfm_tab(self):
        """设置宽场显微镜效果页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        # 输入路径组
        input_group, self.wfm_input_path, self.wfm_input_button = self.create_file_input_group("TIF文件目录：")
        layout.addWidget(input_group)
        
        # 输出路径组
        output_group, self.wfm_output_path, self.wfm_output_button = self.create_file_input_group("输出目录路径：")
        layout.addWidget(output_group)
        
        # 处理按钮
        self.wfm_process_button = QPushButton("批量生成宽场效果")
        self.wfm_process_button.setMinimumHeight(40)
        layout.addWidget(self.wfm_process_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可将三维TIF数据转换为宽场显微镜效果图像。\n"
            "- 自动处理指定目录及其所有子目录中的TIF文件\n"
            "- 使用点扩散函数(PSF)对每一层应用卷积\n"
            "- 保留原始目录结构\n"
            "- 支持处理2D和3D的TIF图像"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "宽场显微镜效果")
        
    def setup_modify_tab(self):
        """设置图像修改页面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        
        # 输入路径组
        input_group, self.modify_input_path, self.modify_input_button = self.create_file_input_group("TIF文件目录：")
        layout.addWidget(input_group)
        
        # 输出路径组
        output_group, self.modify_output_path, self.modify_output_button = self.create_file_input_group("输出目录路径：")
        layout.addWidget(output_group)
        
        # 参数设置组
        params_frame = QFrame()
        params_layout = QHBoxLayout(params_frame)
        params_layout.setContentsMargins(10, 10, 10, 10)
        params_layout.setSpacing(20)
        
        # 旋转角度
        angle_layout = QHBoxLayout()
        angle_label = QLabel("旋转角度：")
        self.angle_spin = QSpinBox()
        self.angle_spin.setRange(-360, 360)
        self.angle_spin.setSingleStep(90)
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spin)
        params_layout.addLayout(angle_layout)
        
        # 裁剪大小
        crop_layout = QHBoxLayout()
        crop_label = QLabel("裁剪大小：")
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(0, 1000)
        self.crop_spin.setSingleStep(10)
        crop_layout.addWidget(crop_label)
        crop_layout.addWidget(self.crop_spin)
        params_layout.addLayout(crop_layout)
        
        # 深度
        depth_layout = QHBoxLayout()
        depth_label = QLabel("深度：")
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100)
        depth_layout.addWidget(depth_label)
        depth_layout.addWidget(self.depth_spin)
        params_layout.addLayout(depth_layout)
        
        layout.addWidget(params_frame)
        
        # 处理按钮
        self.modify_process_button = QPushButton("批量处理图像")
        self.modify_process_button.setMinimumHeight(40)
        layout.addWidget(self.modify_process_button)
        
        # 添加说明
        instructions = QLabel(
            "该功能可批量处理指定目录中的所有TIF文件。\n"
            "- 选择一个选项（旋转角度、裁剪大小或深度）进行处理\n"
            "- 旋转角度：将图像旋转指定的角度\n"
            "- 裁剪大小：裁剪图像到指定大小\n"
            "- 深度：将图像分割为指定数量的层"
        )
        instructions.setStyleSheet("color: #ffffff; background-color: rgba(0, 120, 212, 0.2); padding: 10px; border-radius: 5px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.tabWidget.addTab(tab, "图像修改")
        
    def setup_style(self):
        """设置样式"""
        pass  # 样式已在setupUi中设置 