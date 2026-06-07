import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import json
import math
import sys

# --- logic from hough_rectify_grid.py (integrated) ---
def hough_rectify_grid(input_data, detect_image=None, canny_low=50, canny_high=150, hough_thresh_factor=0.25, blur_margin=100):
    """
    使用霍夫变换检测栅格旋转并纠偏。

    - 边缘检测/角度估计：使用 detect_image（优先使用原图灰度）。
    - 旋转与淡化边缘：作用在传入的 input_data（通常是预处理结果）。
    """
    # 1. 读入目标（用于旋转的预处理图）
    if isinstance(input_data, str):
        if not os.path.exists(input_data): return None, 0
        src = cv2.imdecode(np.fromfile(input_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        src = input_data
    if src is None: return None, 0

    # 1.1 读入检测用原图（仅用于霍夫角度估计）
    if detect_image is None:
        src_for_detect = src.copy()
    else:
        if isinstance(detect_image, str):
            if not os.path.exists(detect_image):
                return src, 0, None
            src_for_detect = cv2.imdecode(np.fromfile(detect_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            src_for_detect = detect_image
        if src_for_detect is None:
            src_for_detect = src.copy()

    h, w = src.shape
    
    # 2. 前处理 - 边缘向黑色过渡 (Fade to black)
    # 为了防止后续SVD检测时出现明显的边缘伪影
    if blur_margin > 0:
        hm, wm = src.shape[:2]
        margin = min(blur_margin, hm//2, wm//2)
        if margin > 0:
            # 创建平滑的过渡掩码 (使用 cos 曲线)
            mask = np.ones((hm, wm), dtype=np.float32)
            grad = 0.5 * (1 - np.cos(np.linspace(0, np.pi, margin, dtype=np.float32)))
            
            # Top
            mask[0:margin, :] *= grad[:, np.newaxis]
            # Bottom
            mask[hm-margin:hm, :] *= grad[::-1, np.newaxis]
            # Left
            mask[:, 0:margin] *= grad[np.newaxis, :]
            # Right
            mask[:, wm-margin:wm] *= grad[np.newaxis, ::-1]
            
            # 将过渡应用到原图
            src = (src.astype(np.float32) * mask).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(src_for_detect)
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    
    # 3. 边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    
    # 4. 霍夫直线变换
    threshold = int(min(w, h) * hough_thresh_factor)
    if threshold <= 0: threshold = 1
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    
    if lines is None:
        return src, 0, None
    
    # 5. 角度分析
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = math.degrees(theta)
        norm_angle = angle % 90
        if norm_angle > 45:
            norm_angle -= 90
        angles.append(norm_angle)
    
    if not angles:
        return src, 0, None
    
    sorted_angles = sorted(angles)
    detected_angle = sorted_angles[len(sorted_angles) // 2]
    
    # 6. 执行纠偏旋转 (Expansion Mode)
    # 计算新的边界框以避免裁剪
    (h, w) = src.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
    
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    
    # 调整变换矩阵的平移分量
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    
    result = cv2.warpAffine(src, M, (bound_w, bound_h), flags=cv2.INTER_LANCZOS4)
    
    return result, detected_angle, M

# --- logic from svd_crack_detector.py (partial) ---
def perform_svd_reconstruction(gray_matrix, k=10):
    """
    执行 SVD 分解并去除前 k 个奇异值进行重建 (背景抑制)
    """
    if gray_matrix is None: return None
    
    # SVD 分解 (Economy)
    U, S, Vt = np.linalg.svd(gray_matrix, full_matrices=False)
    
    # 去除前 k 个主成分
    S_new = S.copy()
    if k < len(S_new):
        S_new[:k] = 0
    else:
        S_new[:] = 0
        
    # 重建
    reconstructed = np.dot(U * S_new, Vt)
    return reconstructed

class IntegratedCrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("太阳能电池裂纹检测集成工作流")
        self.root.geometry("1800x1000")
        try:
            self.root.state('zoomed') # 启动时最大化窗口以获得最高分辨率
        except:
            pass
        
        # --- State ---
        self.current_step = 0
        self.config_path = "crack_detection_config.json"
        self.img_path = None
        
        # Images at each stage
        self.img_step0_original = None
        self.img_step1_tophat = None
        self.img_step2_rectified = None
        self.img_step0_original = None
        self.img_step1_tophat = None
        self.img_step2_rectified = None
        self.img_step3_svd = None
        self.img_step4_filtered = None # 降噪后的二值图
        self.img_restored_overlay = None # 最终形态学/降噪后结果映射回原图
        self.img_restored_cc_overlay = None # 连通域彩色覆盖
        self.img_restored_skeleton_overlay = None # 骨架拟合覆盖
        self.step4_all_executed = False # 标记是否已运行完整后处理流程
        self.display_mode_step4 = tk.StringVar(value="mask_overlay")
        
        self.rotation_matrix = None # 存储旋转矩阵用于反变换
        self._last_step = 0 # 记录上一个步骤以判断导航方向
        self.headless_mode = False # 批处理/无界面时跳过绘制
        
        # --- Parameters (tk variables) ---
        # Step 1: Preprocessing Method Selection
        self.step1_method = tk.StringVar(value="DownsampleBG")  # 默认使用下采样背景归一化
        
        # Black-Hat Parameters
        self.th_kernel_size = tk.IntVar(value=15)
        self.th_kernel_shape = tk.StringVar(value="RECT") # RECT, CROSS, ELLIPSE
        
        # Homomorphic Filter Parameters
        self.homo_gamma_l = tk.DoubleVar(value=0.5)
        self.homo_gamma_h = tk.DoubleVar(value=2.0)
        self.homo_d0 = tk.IntVar(value=30)
        self.homo_c = tk.DoubleVar(value=1.5)

        # Downsample Background Parameters
        self.ds_scale = tk.DoubleVar(value=0.25)
        self.ds_method = tk.StringVar(value="gaussian")
        self.ds_kernel = tk.IntVar(value=41)
        self.ds_offset = tk.DoubleVar(value=10.0)
        self.ds_norm_mode = tk.StringVar(value="divide")  # divide | subtract
        self.ds_invert = tk.BooleanVar(value=False)
        
        # Mask-Blur-Suppress Parameters (from mask_blur_suppress_gui)
        self.mbs_threshold = tk.IntVar(value=240)
        self.mbs_threshold_type = tk.StringVar(value="BINARY")  # BINARY or BINARY_INV
        self.mbs_open_kernel_type = tk.StringVar(value="RECT")  # RECT | ELLIPSE | CROSS
        self.mbs_open_kernel_size = tk.IntVar(value=61)
        self.mbs_dilate_kernel_type = tk.StringVar(value="RECT")  # RECT | ELLIPSE | CROSS
        self.mbs_dilate_kernel_size = tk.IntVar(value=101)
        self.mbs_dilate_iterations = tk.IntVar(value=1)
        self.mbs_blur_kernel = tk.IntVar(value=101)
        self.mbs_edge_smooth = tk.IntVar(value=51)

        # Pre-Opening (before step1 first sub-step)
        self.pre_open_kernel_type = tk.StringVar(value="RECT")
        self.pre_open_kernel_size = tk.IntVar(value=5)
        self.pre_open_iterations = tk.IntVar(value=1)
        
        # Step 2: Hough Rectify
        self.hr_canny_low = tk.IntVar(value=50)
        self.hr_canny_high = tk.IntVar(value=150)
        self.hr_hough_factor = tk.DoubleVar(value=0.13)
        self.hr_blur_margin = tk.IntVar(value=100) # New parameter
        
        # Step 3: SVD
        self.svd_k = tk.IntVar(value=10)
        self.svd_downsample = tk.DoubleVar(value=1.0) # 读入时已统一 0.25，这里默认不再降采样
        
        # Step 4: New Pipeline Parameters
        self.post_log_enable = tk.BooleanVar(value=False)  # Enable LoG in "全部流程"
        self.post_log_sigma = tk.DoubleVar(value=2.0)  # LoG filter sigma
        self.post_log_kernel = tk.IntVar(value=7)  # LoG kernel size
        self.post_median_k = tk.IntVar(value=5)  # Median filter kernel
        self.post_noise_area = tk.IntVar(value=20)  # Remove small area threshold
        self.post_isolated_dist = tk.IntVar(value=20)  # Isolated removal distance
        self.post_dilate_k = tk.IntVar(value=7)  # Dilation kernel size
        self.post_min_area = tk.IntVar(value=20)  # Final filter: min area
        self.post_ar_area_threshold = tk.IntVar(value=200)  # Area override for aspect ratio filter
        self.post_min_ar = tk.DoubleVar(value=2.0)  # Final filter: min aspect ratio
        self.post_max_solidity = tk.DoubleVar(value=0.7)  # Final filter: max solidity
        self.post_large_area_threshold = tk.IntVar(value=100)  # Loose condition: large area override for solidity
        
        # Step 4.2 Auto Binarization Parameters
        self.post_max_components = tk.IntVar(value=50)
        self.post_max_area_ratio = tk.DoubleVar(value=10.0)
        
        # Step 4 State Management
        self.img_step4_current = None  # Current working image in Step 4
        # 当前预览缓存（用于保存预览图像）
        self.current_preview_left = None
        self.current_preview_right = None
        
        # Judgment Parameters
        self.judge_min_count = tk.IntVar(value=1)
        self.judge_min_ratio = tk.DoubleVar(value=0.01) # Percentage 0-100
        self.judge_result_text = tk.StringVar(value="未知")
        self.judge_result_color = "gray"
        # Persisted judgment metrics for batch/reporting
        self.judge_count = 0
        self.judge_ratio = 0.0
        self.judge_defect = None
        
        # Show rejected components option
        self.show_rejected = tk.BooleanVar(value=False)
        self.img_rejected_components = None  # Store rejected components mask
        
        self.steps = [
            "步骤 0: 加载原图",
            "步骤 1: 预处理",
            "步骤 2: 霍夫变换纠偏",
            "步骤 3: SVD 分解 (特征提取)",
            "步骤 4: 形态学处理 (裂纹提取)"
        ]
        
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        # 1. Top Steps Indicator
        self.step_frame = ttk.Frame(self.root, padding=10)
        self.step_frame.pack(side=tk.TOP, fill=tk.X)
        self.step_labels = []
        for i, text in enumerate(self.steps):
            lbl = ttk.Label(self.step_frame, text=text, font=("微软雅黑", 10), foreground="gray")
            lbl.pack(side=tk.LEFT, padx=10)
            self.step_labels.append(lbl)
            
        # 2. Main Work Area
        work_area = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        work_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Controls (scrollable)
        control_outer = ttk.Frame(work_area)
        work_area.add(control_outer, weight=1)
        self.control_canvas = tk.Canvas(control_outer, highlightthickness=0)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.control_scrollbar = ttk.Scrollbar(control_outer, orient=tk.VERTICAL, command=self.control_canvas.yview)
        self.control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        self.control_inner = ttk.Frame(self.control_canvas)
        self.control_window = self.control_canvas.create_window((0, 0), window=self.control_inner, anchor="nw")
        self.control_inner.bind("<Configure>", lambda e: self._on_control_configure())
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfig(self.control_window, width=e.width))
        self._bind_mousewheel(self.control_canvas)
        
        self.control_frame = ttk.LabelFrame(self.control_inner, text="参数配置", padding=15, width=350)
        self.control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Right: Display
        self.display_frame = ttk.LabelFrame(work_area, text="处理预览", padding=5)
        work_area.add(self.display_frame, weight=4)
        
        self.canvas_l = tk.Canvas(self.display_frame, bg="#333")
        self.canvas_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.canvas_r = tk.Canvas(self.display_frame, bg="#333")
        self.canvas_r.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)

        # 3. Bottom Navigation
        self.nav_frame = ttk.Frame(self.root, padding=10)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(self.nav_frame, text="保存默认", command=self._save_config).pack(side=tk.LEFT)
        ttk.Button(self.nav_frame, text="加载默认", command=self._load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="导出配置...", command=self.export_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="导入配置...", command=self.import_config).pack(side=tk.LEFT, padx=5)
        
        self.btn_next = ttk.Button(self.nav_frame, text="下一步 >", command=self.go_next)
        self.btn_next.pack(side=tk.RIGHT)
        self.btn_prev = ttk.Button(self.nav_frame, text="< 上一步", command=self.go_prev)
        self.btn_prev.pack(side=tk.RIGHT, padx=10)
        
        self.lbl_status = ttk.Label(self.nav_frame, text="就绪")
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # Initialize UI for step 0
        self.update_ui_state()

    def _on_control_configure(self):
        if hasattr(self, "control_canvas"):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def _bind_mousewheel(self, widget):
        # Enable vertical scroll with mouse wheel
        def _on_mousewheel(event):
            delta = event.delta
            if sys.platform == "darwin":
                widget.yview_scroll(int(-1 * delta), "units")
            else:
                widget.yview_scroll(int(-1 * (delta / 120)), "units")
        widget.bind_all("<MouseWheel>", _on_mousewheel)

    def clear_control_panel(self):
        for widget in self.control_frame.winfo_children():
            widget.destroy()
            
    def update_ui_state(self):
        previous_step = getattr(self, "_last_step", 0)
        # Update Step Indicator
        for i, lbl in enumerate(self.step_labels):
            if i == self.current_step:
                lbl.config(foreground="blue", font=("微软雅黑", 12, "bold"))
            else:
                lbl.config(foreground="gray", font=("微软雅黑", 10))

        # Re-build controls for current step
        self.clear_control_panel()
        
        if self.current_step == 0:
            self._ui_step0()
        elif self.current_step == 1:
            self._ui_step1()
        elif self.current_step == 2:
            self._ui_step2()
        elif self.current_step == 3:
            self._ui_step3()
        elif self.current_step == 4:
            if previous_step != 4:
                self.step4_all_executed = False
            self._ui_step4()
            
        # 逻辑：如果从正向（步骤2 -> 3）进入 SVD，则自动处理
        # 如果从反向（步骤4 -> 3）进入，则不自动处理，保留之前的计算结果
        should_process = True
        if self.current_step == 3:
            if previous_step == 2:
                should_process = True
            else:
                # 如果是从其它步骤（如4）回退回来的，或者当前已有结果，则不自动重算
                should_process = False if self.img_step3_svd is not None else True

        self._last_step = self.current_step
        self.update_display(process_current=should_process)

        # 确保底部导航栏始终可见
        if hasattr(self, "nav_frame"):
            try:
                self.nav_frame.pack_forget()
                self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
                self.nav_frame.lift()
                self.root.update_idletasks()
            except Exception:
                pass
        
        # 滚动容器重置到顶部，避免因长控件导致底部按钮被挤出可视区
        if hasattr(self, "control_canvas"):
            try:
                self.control_canvas.yview_moveto(0)
            except Exception:
                pass
            
    # --- Step UI Builders ---
    def _ui_step0(self):
        ttk.Button(self.control_frame, text="选择图片...", command=self.load_image).pack(fill=tk.X, pady=20)
        ttk.Label(self.control_frame, text="请先加载一张图片以开始工作流程。", wraplength=200).pack(pady=10)

    def _ui_step1(self):
        # 前置开运算（自动执行）
        pre_open = ttk.LabelFrame(self.control_frame, text="前置开运算（自动执行）", padding=5)
        pre_open.pack(fill=tk.X, pady=(0, 10))

        row1 = ttk.Frame(pre_open); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="结构元:").pack(side=tk.LEFT)
        ttk.Combobox(row1, values=["RECT", "ELLIPSE", "CROSS"], textvariable=self.pre_open_kernel_type, state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1, text="尺寸(奇数):").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(row1, from_=1, to=201, increment=2, textvariable=self.pre_open_kernel_size, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(row1, text="迭代:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(row1, from_=1, to=10, textvariable=self.pre_open_iterations, width=4).pack(side=tk.LEFT, padx=5)

        ttk.Button(pre_open, text="更新预览", command=lambda: self.update_display()).pack(fill=tk.X, pady=(6,0))

        ttk.Label(self.control_frame, text="预处理方法选择", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, pady=(10, 10))
        
        # Method Selection
        method_frame = ttk.Frame(self.control_frame)
        method_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(method_frame, text="底帽变换 (Black-Hat)", variable=self.step1_method, value="BlackHat", command=self.update_ui_state).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="同态滤波 (Homomorphic)", variable=self.step1_method, value="Homomorphic", command=self.update_ui_state).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="下采样背景归一化", variable=self.step1_method, value="DownsampleBG", command=self.update_ui_state).pack(side=tk.LEFT, padx=5)
        
        # Black-Hat Parameters
        if self.step1_method.get() == "BlackHat":
            bh_frame = ttk.LabelFrame(self.control_frame, text="底帽变换参数", padding=5)
            bh_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(bh_frame, text="结构元素大小:").pack(anchor=tk.W)
            ttk.Scale(bh_frame, from_=1, to=400, variable=self.th_kernel_size, command=lambda e: self.update_display()).pack(fill=tk.X)
            ttk.Label(bh_frame, textvariable=self.th_kernel_size).pack(anchor=tk.E)
            
            ttk.Label(bh_frame, text="形状:").pack(anchor=tk.W, pady=(10, 0))
            shape_cb = ttk.Combobox(bh_frame, values=["RECT", "CROSS", "ELLIPSE"], textvariable=self.th_kernel_shape, state="readonly")
            shape_cb.pack(fill=tk.X)
            shape_cb.bind("<<ComboboxSelected>>", lambda e: self.update_display())
        
        elif self.step1_method.get() == "DownsampleBG":
            ds_frame = ttk.LabelFrame(self.control_frame, text="下采样背景归一化参数", padding=5)
            ds_frame.pack(fill=tk.X, pady=10)

            # Scale
            ttk.Label(ds_frame, text="下采样比例 (0-1)").pack(anchor=tk.W)
            ttk.Scale(ds_frame, from_=0.05, to=0.95, variable=self.ds_scale).pack(fill=tk.X)
            ttk.Label(ds_frame, textvariable=self.ds_scale).pack(anchor=tk.E)

            # Kernel
            k_frame = ttk.Frame(ds_frame)
            k_frame.pack(fill=tk.X, pady=4)
            ttk.Label(k_frame, text="滤波核 (odd)").pack(side=tk.LEFT)
            ttk.Spinbox(k_frame, from_=3, to=201, increment=2, textvariable=self.ds_kernel, width=8).pack(side=tk.RIGHT)

            # Method
            m_frame = ttk.Frame(ds_frame)
            m_frame.pack(fill=tk.X, pady=4)
            ttk.Label(m_frame, text="滤波方式").pack(side=tk.LEFT)
            ttk.Combobox(m_frame, values=["gaussian", "median", "opening"], textvariable=self.ds_method, state="readonly").pack(side=tk.RIGHT, fill=tk.X, expand=True)

            # Norm mode
            norm_frame = ttk.LabelFrame(ds_frame, text="归一化方式", padding=4)
            norm_frame.pack(fill=tk.X, pady=4)
            ttk.Radiobutton(norm_frame, text="除法 (推荐)", variable=self.ds_norm_mode, value="divide").pack(anchor=tk.W)
            ttk.Radiobutton(norm_frame, text="减法", variable=self.ds_norm_mode, value="subtract").pack(anchor=tk.W)

            # Offset for subtract
            off_frame = ttk.Frame(ds_frame)
            off_frame.pack(fill=tk.X, pady=4)
            ttk.Label(off_frame, text="减法偏移").pack(side=tk.LEFT)
            ttk.Entry(off_frame, textvariable=self.ds_offset, width=8).pack(side=tk.RIGHT)

            # Invert option
            ttk.Checkbutton(ds_frame, text="输出取反", variable=self.ds_invert).pack(anchor=tk.W, pady=4)

            ttk.Button(ds_frame, text="应用下采样归一化", command=lambda: self.update_display()).pack(fill=tk.X, pady=(6,0))

        

        # Homomorphic Filter Parameters
        else:
            homo_frame = ttk.LabelFrame(self.control_frame, text="同态滤波参数", padding=5)
            homo_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(homo_frame, text="Gamma Low (暗部增强):").pack(anchor=tk.W)
            ttk.Entry(homo_frame, textvariable=self.homo_gamma_l).pack(fill=tk.X)
            
            ttk.Label(homo_frame, text="Gamma High (边缘增强):").pack(anchor=tk.W, pady=(5,0))
            ttk.Entry(homo_frame, textvariable=self.homo_gamma_h).pack(fill=tk.X)
            
            ttk.Label(homo_frame, text="Cutoff (截止频率):").pack(anchor=tk.W, pady=(5,0))
            ttk.Entry(homo_frame, textvariable=self.homo_d0).pack(fill=tk.X)
            
            ttk.Label(homo_frame, text="Slope (斜率):").pack(anchor=tk.W, pady=(5,0))
            ttk.Entry(homo_frame, textvariable=self.homo_c).pack(fill=tk.X)
            
            ttk.Button(homo_frame, text="应用滤波", command=lambda: self.update_display()).pack(fill=tk.X, pady=(10,0))

        # 掩模模糊抑制参数（自动执行）
        mbs = ttk.LabelFrame(self.control_frame, text="掩模模糊抑制参数（自动执行）", padding=5)
        mbs.pack(fill=tk.X, pady=10)

        # Threshold
        thf = ttk.Frame(mbs)
        thf.pack(fill=tk.X, pady=2)
        ttk.Label(thf, text="阈值:").pack(side=tk.LEFT)
        ttk.Spinbox(thf, from_=0, to=255, textvariable=self.mbs_threshold, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(thf, text="类型:").pack(side=tk.LEFT, padx=(10, 0))
        ttk.Combobox(thf, values=["BINARY", "BINARY_INV"], textvariable=self.mbs_threshold_type, state="readonly", width=12).pack(side=tk.LEFT)

        # Opening
        opf = ttk.LabelFrame(mbs, text="开运算", padding=4)
        opf.pack(fill=tk.X, pady=4)
        of1 = ttk.Frame(opf); of1.pack(fill=tk.X, pady=2)
        ttk.Label(of1, text="结构元:").pack(side=tk.LEFT)
        ttk.Combobox(of1, values=["RECT", "ELLIPSE", "CROSS"], textvariable=self.mbs_open_kernel_type, state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(of1, text="尺寸(奇数):").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(of1, from_=1, to=301, increment=2, textvariable=self.mbs_open_kernel_size, width=6).pack(side=tk.LEFT, padx=5)

        # Dilation
        dlf = ttk.LabelFrame(mbs, text="膨胀", padding=4)
        dlf.pack(fill=tk.X, pady=4)
        df1 = ttk.Frame(dlf); df1.pack(fill=tk.X, pady=2)
        ttk.Label(df1, text="结构元:").pack(side=tk.LEFT)
        ttk.Combobox(df1, values=["RECT", "ELLIPSE", "CROSS"], textvariable=self.mbs_dilate_kernel_type, state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(df1, text="尺寸(奇数):").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(df1, from_=1, to=401, increment=2, textvariable=self.mbs_dilate_kernel_size, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(df1, text="迭代:").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(df1, from_=1, to=10, textvariable=self.mbs_dilate_iterations, width=4).pack(side=tk.LEFT, padx=5)

        # Blur
        blf = ttk.LabelFrame(mbs, text="模糊与边缘平滑", padding=4)
        blf.pack(fill=tk.X, pady=4)
        bf1 = ttk.Frame(blf); bf1.pack(fill=tk.X, pady=2)
        ttk.Label(bf1, text="高斯核(奇数):").pack(side=tk.LEFT)
        ttk.Spinbox(bf1, from_=1, to=401, increment=2, textvariable=self.mbs_blur_kernel, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(bf1, text="边缘平滑(奇数):").pack(side=tk.LEFT, padx=(10,0))
        ttk.Spinbox(bf1, from_=1, to=201, increment=2, textvariable=self.mbs_edge_smooth, width=6).pack(side=tk.LEFT, padx=5)

        ttk.Button(mbs, text="更新预览", command=lambda: self.update_display()).pack(fill=tk.X, pady=(6,0))

    def _ui_step2(self):
        ttk.Label(self.control_frame, text="霍夫变换纠偏参数").pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Label(self.control_frame, text="Canny 低阈值:").pack(anchor=tk.W)
        ttk.Scale(self.control_frame, from_=10, to=200, variable=self.hr_canny_low).pack(fill=tk.X)
        ttk.Label(self.control_frame, textvariable=self.hr_canny_low).pack(anchor=tk.E)
        
        ttk.Label(self.control_frame, text="Canny 高阈值:").pack(anchor=tk.W)
        ttk.Scale(self.control_frame, from_=50, to=400, variable=self.hr_canny_high).pack(fill=tk.X)
        ttk.Label(self.control_frame, textvariable=self.hr_canny_high).pack(anchor=tk.E)
        
        ttk.Label(self.control_frame, text="霍夫投票系数 (0.01-0.5):").pack(anchor=tk.W)
        ttk.Scale(self.control_frame, from_=0.01, to=0.5, variable=self.hr_hough_factor).pack(fill=tk.X)
        ttk.Label(self.control_frame, textvariable=self.hr_hough_factor).pack(anchor=tk.E)

        ttk.Label(self.control_frame, text="边缘模糊范围 (像素):").pack(anchor=tk.W)
        ttk.Scale(self.control_frame, from_=0, to=300, variable=self.hr_blur_margin).pack(fill=tk.X)
        ttk.Label(self.control_frame, textvariable=self.hr_blur_margin).pack(anchor=tk.E)
        
        ttk.Button(self.control_frame, text="执行纠偏", command=lambda: self.update_display(process_current=True)).pack(fill=tk.X, pady=20)

    def _ui_step3(self):
        ttk.Label(self.control_frame, text="SVD 分解参数 (性能优化版)").pack(anchor=tk.W, pady=(0, 10))
        
        # 降采样选择
        ttk.Label(self.control_frame, text="处理分辨率 (降采样):").pack(anchor=tk.W)
        ds_frame = ttk.Frame(self.control_frame)
        ds_frame.pack(fill=tk.X, pady=5)
        for val, text in [(1.0, "原图"), (0.5, "0.5x"), (0.25, "0.25x")]:
            ttk.Radiobutton(ds_frame, text=text, variable=self.svd_downsample, value=val).pack(side=tk.LEFT, padx=5)

        ttk.Label(self.control_frame, text="去除前 K 个奇异值:").pack(anchor=tk.W, pady=(10, 0))
        k_frame = ttk.Frame(self.control_frame)
        k_frame.pack(fill=tk.X)
        ttk.Entry(k_frame, textvariable=self.svd_k).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(self.control_frame, text="开始 SVD 计算", command=lambda: self.update_display(process_current=True)).pack(fill=tk.X, pady=10)
        
        ttk.Label(self.control_frame, text="注意：4K 图像建议选用 0.5x，K 值越大去背景越强。", wraplength=250, foreground="gray").pack(pady=10)

    def _ui_step4(self):
        ttk.Label(self.control_frame, text="新型后处理流程 (SVD → 输出)", font=("微软雅黑", 10, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        # === Reset Button ===
        ttk.Button(self.control_frame, text="🔄重置步骤4（回到SVD输出）", command=self.reset_step4).pack(fill=tk.X, pady=5)
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # === Step 4.0.5: LoG Filter ===
        log_frame = ttk.LabelFrame(self.control_frame, text="4.0.5 LoG 算子滤波", padding="5")
        log_frame.pack(fill=tk.X, pady=3)
        
        # Enable LoG in full pipeline
        ttk.Checkbutton(log_frame, text="启用并纳入“全部流程”", variable=self.post_log_enable).pack(anchor=tk.W)
        
        # Sigma parameter
        sigma_frame = ttk.Frame(log_frame)
        sigma_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sigma_frame, text="Sigma (标准差):").pack(side=tk.LEFT)
        ttk.Spinbox(sigma_frame, from_=0.5, to=10.0, increment=0.1, textvariable=self.post_log_sigma, width=6).pack(side=tk.LEFT, padx=5)
        
        # Kernel size parameter
        kernel_frame = ttk.Frame(log_frame)
        kernel_frame.pack(fill=tk.X, pady=2)
        ttk.Label(kernel_frame, text="卷积核大小:").pack(side=tk.LEFT)
        ttk.Spinbox(kernel_frame, from_=3, to=31, increment=2, textvariable=self.post_log_kernel, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(log_frame, text="执行 LoG 滤波", command=self.apply_step4_log).pack(fill=tk.X, pady=2)
        
        # === Step 4.1: Median Blur ===
        med_frame = ttk.LabelFrame(self.control_frame, text="4.1 中值滤波", padding="5")
        med_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(med_frame, text="卷积核大小:").pack(side=tk.LEFT)
        ttk.Spinbox(med_frame, from_=3, to=15, increment=2, textvariable=self.post_median_k, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(med_frame, text="执行中值滤波", command=self.apply_step4_median).pack(side=tk.RIGHT, padx=2)
        
        
        # === Step 4.2: Auto Binarization ===
        bin_frame = ttk.LabelFrame(self.control_frame, text="4.2 自动二值化 (自适应)", padding="5")
        bin_frame.pack(fill=tk.X, pady=3)
        
        # Max Components
        comp_frame = ttk.Frame(bin_frame)
        comp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(comp_frame, text="最大连通分量数:").pack(side=tk.LEFT)
        ttk.Spinbox(comp_frame, from_=5, to=500, increment=5, textvariable=self.post_max_components, width=8).pack(side=tk.RIGHT)
        
        # Max Area Ratio
        area_frame = ttk.Frame(bin_frame)
        area_frame.pack(fill=tk.X, pady=2)
        ttk.Label(area_frame, text="最大前景面积比例(%):").pack(side=tk.LEFT)
        ttk.Spinbox(area_frame, from_=0.1, to=50.0, increment=0.5, textvariable=self.post_max_area_ratio, width=8).pack(side=tk.RIGHT)
        
        ttk.Button(bin_frame, text="执行自适应二值化", command=self.apply_step4_binary).pack(fill=tk.X, pady=(5,0))
        
        # === Step 4.3: Remove Small Area ===
        small_frame = ttk.LabelFrame(self.control_frame, text="4.3 删除小面积噪点", padding="5")
        small_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(small_frame, text="最大噪点面积:").pack(side=tk.LEFT)
        ttk.Spinbox(small_frame, from_=1, to=200, textvariable=self.post_noise_area, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Button(small_frame, text="删除小面积", command=self.apply_step4_remove_small).pack(side=tk.RIGHT, padx=2)
        
        # === Step 4.4: Remove Isolated ===
        iso_frame = ttk.LabelFrame(self.control_frame, text="4.4 删除孤立点", padding="5")
        iso_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(iso_frame, text="最大邻域距离:").pack(side=tk.LEFT)
        ttk.Spinbox(iso_frame, from_=1, to=100, textvariable=self.post_isolated_dist, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Button(iso_frame, text="删除孤立点", command=self.apply_step4_remove_isolated).pack(side=tk.RIGHT, padx=2)
        
        # === Step 4.5: Dilation ===
        dilate_frame = ttk.LabelFrame(self.control_frame, text="4.5 膨胀", padding="5")
        dilate_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(dilate_frame, text="膨胀核大小:").pack(side=tk.LEFT)
        ttk.Spinbox(dilate_frame, from_=1, to=21, increment=2, textvariable=self.post_dilate_k, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(dilate_frame, text="执行膨胀", command=self.apply_step4_dilate).pack(side=tk.RIGHT, padx=2)
        
        # === Step 4.6: Final Filter ===
        filter_frame = ttk.LabelFrame(self.control_frame, text="4.6 连通域筛选", padding="5")
        filter_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(filter_frame, text="最小面积:").pack(anchor=tk.W)
        ttk.Scale(filter_frame, from_=1, to=1000, variable=self.post_min_area).pack(fill=tk.X)
        ttk.Label(filter_frame, textvariable=self.post_min_area).pack(anchor=tk.E)

        ttk.Label(filter_frame, text="长宽比豁免面积阈值:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(filter_frame, from_=1, to=2000, variable=self.post_ar_area_threshold).pack(fill=tk.X)
        ttk.Label(filter_frame, textvariable=self.post_ar_area_threshold).pack(anchor=tk.E)
        
        ttk.Label(filter_frame, text="最小长宽比:").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(filter_frame, from_=1.0, to=10.0, variable=self.post_min_ar).pack(fill=tk.X)
        ttk.Label(filter_frame, textvariable=self.post_min_ar).pack(anchor=tk.E)

        ttk.Label(filter_frame, text="最大实心度 (Solidity):").pack(anchor=tk.W, pady=(5, 0))
        ttk.Scale(filter_frame, from_=0.1, to=1.0, variable=self.post_max_solidity).pack(fill=tk.X)
        ttk.Label(filter_frame, textvariable=self.post_max_solidity).pack(anchor=tk.E)
        
        ttk.Button(filter_frame, text="执行连通域筛选", command=self.apply_step4_filter).pack(fill=tk.X, pady=5)

        # === Execute All ===
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(self.control_frame, text="⚡ 执行全部流程 (LoG可选 → 4.1 → 4.6)", 
               command=self.execute_step4_all).pack(fill=tk.X, pady=5)

        # 保存当前预览窗口图像（右侧结果）
        ttk.Button(self.control_frame, text="保存当前预览图像", command=self.save_current_preview).pack(fill=tk.X, pady=5)

        # 结果显示模式
        mode_frame = ttk.LabelFrame(self.control_frame, text="结果显示模式", padding="5")
        mode_frame.pack(fill=tk.X, pady=5)
        for text, val in [("连通域覆盖", "cc_overlay"), ("骨架拟合覆盖", "skeleton_overlay"), ("二值/掩膜", "mask_overlay")]:
            ttk.Radiobutton(mode_frame, text=text, value=val, variable=self.display_mode_step4,
                            command=lambda: self.update_display(process_current=False)).pack(anchor=tk.W)
        
        # 显示被剔除的连通分量（仅在二值/掩膜模式生效）
        ttk.Checkbutton(mode_frame, text="显示被剔除的连通分量", variable=self.show_rejected,
                       command=lambda: self.update_display(process_current=False)).pack(anchor=tk.W, pady=(5,0))

        # --- Judgment Criteria ---
        judge_frame = ttk.LabelFrame(self.control_frame, text="判定标准", padding="5")
        judge_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(judge_frame, text="判定存在裂纹的条件 (且):", font=("微软雅黑", 9, "bold")).pack(anchor=tk.W)
        
        f1 = ttk.Frame(judge_frame)
        f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="最小数量 >=").pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.judge_min_count, width=5).pack(side=tk.RIGHT)
        
        f2 = ttk.Frame(judge_frame)
        f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="或 面积占比(%) >=").pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.judge_min_ratio, width=8).pack(side=tk.RIGHT)
        
        # Result Label
        self.lbl_judge = ttk.Label(self.control_frame, textvariable=self.judge_result_text, font=("微软雅黑", 16, "bold"), anchor="center")
        self.lbl_judge.pack(fill=tk.X, pady=15)
        
        ttk.Button(self.control_frame, text="更新判定", command=lambda: self.update_display()).pack(fill=tk.X)

        ttk.Button(self.control_frame, text="保存所有结果", command=self.save_all_results).pack(fill=tk.X, pady=20)


    # --- Logic & Display ---
    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.step4_all_executed = False
            # Load gray
            self.img_path = path
            self.img_step0_original = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            # 统一将输入缩放到 0.25 倍（符合批处理脚本）
            if self.img_step0_original is not None:
                h, w = self.img_step0_original.shape[:2]
                nh, nw = max(1, int(h * 0.25)), max(1, int(w * 0.25))
                self.img_step0_original = cv2.resize(self.img_step0_original, (nw, nh), interpolation=cv2.INTER_AREA)
                if hasattr(self, 'lbl_status'):
                    self.lbl_status.config(text=f"已按 0.25 缩放输入，尺寸: {w}x{h} → {nw}x{nh}")
            self.update_display()

    def process_step(self, step_idx):
        """Execute logic for a specific step"""
        if step_idx == 0:
            pass # No processing
            
        elif step_idx == 1:
            # Step 1: 前置开运算 -> 掩模模糊抑制 -> 亮度均一化（三选一）
            if self.img_step0_original is None: return
            method = self.step1_method.get()

            # 先做前置开运算（在原图上）
            ksize = int(self.pre_open_kernel_size.get())
            if ksize % 2 == 0:
                ksize += 1
            pre_kernel = self._get_kernel(self.pre_open_kernel_type.get(), ksize)
            pre_iter = max(1, int(self.pre_open_iterations.get()))
            pre_open_out = cv2.morphologyEx(self.img_step0_original, cv2.MORPH_OPEN, pre_kernel, iterations=pre_iter)

            # 再做掩模模糊抑制（在开运算输出上）
            mbs_out = self.apply_mask_blur_suppress(
                pre_open_out,
                threshold=int(self.mbs_threshold.get()),
                threshold_type=self.mbs_threshold_type.get(),
                open_kernel_type=self.mbs_open_kernel_type.get(),
                open_kernel_size=int(self.mbs_open_kernel_size.get()),
                dilate_kernel_type=self.mbs_dilate_kernel_type.get(),
                dilate_kernel_size=int(self.mbs_dilate_kernel_size.get()),
                dilate_iterations=int(self.mbs_dilate_iterations.get()),
                blur_kernel=int(self.mbs_blur_kernel.get()),
                edge_smooth=int(self.mbs_edge_smooth.get())
            )

            # 再进行所选的亮度均一化
            if method == "BlackHat":
                ks = self.th_kernel_size.get()
                shape_map = {"RECT": cv2.MORPH_RECT, "CROSS": cv2.MORPH_CROSS, "ELLIPSE": cv2.MORPH_ELLIPSE}
                kernel = cv2.getStructuringElement(shape_map[self.th_kernel_shape.get()], (ks, ks))
                self.img_step1_tophat = cv2.morphologyEx(mbs_out, cv2.MORPH_BLACKHAT, kernel)
            elif method == "DownsampleBG":
                scale = float(self.ds_scale.get())
                scale = min(max(scale, 0.01), 0.99)
                k = int(self.ds_kernel.get())
                if k % 2 == 0:
                    k += 1
                bg = self.estimate_background_downsample(
                    mbs_out,
                    scale=scale,
                    method=self.ds_method.get(),
                    kernel=k
                )
                if self.ds_norm_mode.get() == "subtract":
                    norm = self.normalize_subtract(mbs_out, bg, offset=float(self.ds_offset.get()))
                else:
                    norm = self.normalize_divide(mbs_out, bg)
                if self.ds_invert.get():
                    norm = 255 - norm
                self.img_step1_tophat = norm
            else:  # Homomorphic
                self.img_step1_tophat = self.apply_homomorphic_filter(
                    mbs_out,
                    self.homo_gamma_l.get(),
                    self.homo_gamma_h.get(),
                    self.homo_d0.get(),
                    self.homo_c.get()
                )
            
        elif step_idx == 2:
            # Rectify
            inp = self.img_step1_tophat if self.img_step1_tophat is not None else self.img_step0_original
            detect_ref = self.img_step0_original if self.img_step0_original is not None else inp
            if inp is None: return
            rectified, angle, M = hough_rectify_grid(inp,
                                                  detect_image=detect_ref,
                                                  canny_low=self.hr_canny_low.get(), 
                                                  canny_high=self.hr_canny_high.get(), 
                                                  hough_thresh_factor=self.hr_hough_factor.get(),
                                                  blur_margin=self.hr_blur_margin.get())
            self.img_step2_rectified = rectified
            self.rotation_matrix = M
            self.lbl_status.config(text=f"检测到角度: {angle:.2f}° (画布已扩展)")
            
        elif step_idx == 3:
            # SVD
            inp = self.img_step2_rectified
            if inp is None: return
            
            # --- 降采样策略 ---
            factor = self.svd_downsample.get()
            if factor < 0.99: # 考虑浮点误差
                h, w = inp.shape[:2]
                inp_processed = cv2.resize(inp, (int(w*factor), int(h*factor)), interpolation=cv2.INTER_AREA)
            else:
                inp_processed = inp
                
            self.lbl_status.config(text="SVD 计算中，请稍候...")
            self.root.update_idletasks() # 强制刷新 UI 显示状态
            
            recon = perform_svd_reconstruction(inp_processed.astype(float), self.svd_k.get())
            # Normalize to 0-255 for visualization and next step
            if recon is not None:
                norm = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX)
                self.img_step3_svd = norm.astype(np.uint8)
            self.lbl_status.config(text="SVD 计算完成")
                
        elif step_idx == 4:
            # Step 4 is now skipped - binarization is part of Step 5 pipeline
            pass
            
        elif step_idx == 5:
            # Step 5 processing is handled by individual step methods
            pass

    def update_display(self, process_current=True):
        if process_current:
            self.process_step(self.current_step)
        if self.headless_mode:
            return
        
        # Left Canvas: Input for current step
        # Right Canvas: Output for current step
        
        img_l, img_r = None, None
        title_l, title_r = "", ""
        
        if self.current_step == 0:
            img_l = self.img_step0_original
            title_l = "原始图像"
            img_r = self.img_step0_original
            title_r = "（准备开始）"
            
        elif self.current_step == 1:
            img_l = self.img_step0_original
            title_l = "输入: 原图"
            img_r = self.img_step1_tophat
            # 动态标题：开运算 + 掩模抑制 + 所选方法
            method_map = {
                "BlackHat": "开运算 + 掩模抑制 + 底帽变换",
                "Homomorphic": "开运算 + 掩模抑制 + 同态滤波",
                "DownsampleBG": "开运算 + 掩模抑制 + 下采样归一化",
            }
            title_r = f"输出: {method_map.get(self.step1_method.get(), '开运算 + 掩模抑制 + 预处理')} 结果"
            
        elif self.current_step == 2:
            img_l = self.img_step1_tophat
            title_l = "输入: 预处理结果"
            img_r = self.img_step2_rectified
            title_r = "输出: 纠偏结果"
            
        elif self.current_step == 3:
            img_l = self.img_step2_rectified
            title_l = "输入: 纠偏结果"
            img_r = self.img_step3_svd
            title_r = "输出: SVD 背景抑制"

        elif self.current_step == 4:
            # Left: Show SVD input
            img_l = self.img_step3_svd
            title_l = "输入: SVD 结果"
            
            # Right: Show current pipeline result or final filtered result
            if self.img_step4_filtered is not None:
                mode = self.display_mode_step4.get()
                if mode == "cc_overlay" and self.img_restored_cc_overlay is not None:
                    img_r = self.img_restored_cc_overlay
                    title_r = "连通域覆盖叠加 (原图)"
                elif mode == "skeleton_overlay" and self.img_restored_skeleton_overlay is not None:
                    img_r = self.img_restored_skeleton_overlay
                    title_r = "骨架拟合叠加 (原图)"
                elif mode == "mask_overlay" and self.img_restored_overlay is not None:
                    # Check if we should show rejected components
                    if self.show_rejected.get() and self.img_rejected_components is not None:
                        # Create overlay with rejected components in different color
                        img_r = self._create_rejected_overlay()
                        title_r = "最终结果叠加 (红色:保留 蓝色:剔除)"
                    else:
                        img_r = self.img_restored_overlay
                        title_r = "最终结果叠加 (已还原至原图)"
                else:
                    img_r = self.img_step4_filtered
                    title_r = "最终筛选结果"
            elif self.img_step4_current is not None:
                img_r = self.img_step4_current
                title_r = "当前处理结果"
            else:
                img_r = self.img_step3_svd
                title_r = "点击执行按钮开始处理"

            # 记录当前预览用于保存
            self.current_preview_left = img_l
            self.current_preview_right = img_r

        self._show_on_canvas(self.canvas_l, img_l, title_l)
        self._show_on_canvas(self.canvas_r, img_r, title_r, is_result=True)

    def _show_on_canvas(self, canvas, cv_img, title, is_result=False):
        if self.headless_mode:
            return
        canvas.delete("all")
        if cv_img is None: return
        
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10: cw = 400
        if ch < 10: ch = 400
        
        h, w = cv_img.shape[:2]
        scale = min(cw/w, ch/h) * 0.95
        nw, nh = int(w*scale), int(h*scale)
        
        resized = cv2.resize(cv_img, (nw, nh))
        
        # Handle both grayscale and color images
        if len(resized.shape) == 3:
            # Color image (BGR) - convert to RGB for PIL
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        canvas.create_image(cw//2, ch//2, image=tk_img)
        color = "lime" if is_result else "white"
        canvas.create_text(cw//2, 20, text=title, fill=color, font=("微软雅黑", 14, "bold"))
        canvas.image = tk_img # keep ref

    def save_current_preview(self):
        """保存当前步骤右侧预览图像（步骤4专用）。"""
        if self.headless_mode:
            return
        if self.current_step != 4:
            messagebox.showwarning("提示", "请在步骤4中使用此功能")
            return
        img = self.current_preview_right if self.current_preview_right is not None else self.current_preview_left
        if img is None:
            messagebox.showwarning("提示", "当前无可保存的预览图像")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif;*.tiff")],
            title="保存预览图像"
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            # 编码并保存，兼容中文路径
            encode_ext = ".png"
            if ext in [".jpg", ".jpeg"]:
                encode_ext = ".jpg"
            elif ext == ".bmp":
                encode_ext = ".bmp"
            elif ext in [".tif", ".tiff"]:
                encode_ext = ".tiff"
            success, buf = cv2.imencode(encode_ext, img)
            if not success:
                raise RuntimeError("图像编码失败")
            buf.tofile(path)
            messagebox.showinfo("保存成功", f"预览图像已保存至:\n{path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存图像: {e}")

    def go_next(self):
        if self.current_step == 4:
            if not getattr(self, "step4_all_executed", False):
                self.lbl_status.config(text="正在自动执行完整后处理流程...")
                self.execute_step4_all(auto_trigger=True)
                return
            self.save_all_results()
            self.current_step = 0
            self.step4_all_executed = False
            self.update_ui_state()
            return

        if self.current_step < 4:
            if self.current_step == 0 and self.img_step0_original is None:
                messagebox.showwarning("提示", "请先加载图片！")
                return
            if self.current_step == 3 and self.img_step3_svd is None:
                messagebox.showwarning("提示", "请先点击‘开始 SVD 计算’完成特征提取后再前往下一步。")
                return
            self.current_step += 1
            self.update_ui_state()

    def go_prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_ui_state()
            
    def save_all_results(self, out_root=None, base_name=None, quiet=False, save_keys=None, flatten=False):
        if not self.img_path and not base_name:
            return
        out_root = out_root or "final_results"
        if not os.path.exists(out_root):
            os.makedirs(out_root)
            
        base = base_name or os.path.splitext(os.path.basename(self.img_path))[0]
        # Create subfolder for this image unless flatten requested
        if flatten:
            out_dir = out_root
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(out_root, base)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
        # Save selectable steps
        items = {
            "0_original": (self.img_step0_original, "0_original.jpg"),
            "1_blackhat": (self.img_step1_tophat, "1_blackhat.jpg"),
            "2_rectified": (self.img_step2_rectified, "2_rectified.jpg"),
            "3_svd": (self.img_step3_svd, "3_svd_recon.jpg"),
            "4_binary": (self.img_step4_filtered, "4_morph_binary.png"),
            "4_filtered": (getattr(self, "current_step4_result", None), "4_final_filtered_binary.png"),
            "5_overlay": (self.img_restored_overlay, "5_final_overlay.jpg"),
            "5_cc": (self.img_restored_cc_overlay, "5_cc_overlay.jpg"),
            "5_skeleton": (self.img_restored_skeleton_overlay, "5_skeleton_overlay.jpg"),
        }
        allow = set(save_keys) if save_keys else set(items.keys())
        for key, (img, fname) in items.items():
            if key in allow and img is not None:
                if flatten and base:
                    target_name = f"{base}_{fname}"
                else:
                    target_name = fname
                cv2.imwrite(os.path.join(out_dir, target_name), img)

        # Persist detection metrics for reporting
        metrics = {
            "count": self.judge_count,
            "ratio": self.judge_ratio,
            "is_defect": bool(self.judge_defect) if self.judge_defect is not None else None,
            "judge_text": self.judge_result_text.get(),
            "judge_color": self.judge_result_color,
        }
        metrics_path = os.path.join(out_dir, f"{base}_metrics.json") if flatten and base else os.path.join(out_dir, "metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
            
        if not quiet:
            messagebox.showinfo("完成", f"所有处理结果已保存至:\n{os.path.abspath(out_dir)}")
        return out_dir

    def get_config_dict(self):
        """Collect all parameters into a dictionary."""
        return {
            "step1_method": self.step1_method.get(),
            # Pre-Opening
            "pre_open_kernel_type": self.pre_open_kernel_type.get(),
            "pre_open_kernel_size": self.pre_open_kernel_size.get(),
            "pre_open_iterations": self.pre_open_iterations.get(),
            "th_kernel_size": self.th_kernel_size.get(),
            "th_kernel_shape": self.th_kernel_shape.get(),
            "homo_gamma_l": self.homo_gamma_l.get(),
            "homo_gamma_h": self.homo_gamma_h.get(),
            "homo_d0": self.homo_d0.get(),
            "homo_c": self.homo_c.get(),
            "ds_scale": self.ds_scale.get(),
            "ds_method": self.ds_method.get(),
            "ds_kernel": self.ds_kernel.get(),
            "ds_offset": self.ds_offset.get(),
            "ds_norm_mode": self.ds_norm_mode.get(),
            "ds_invert": self.ds_invert.get(),
            # Mask-Blur-Suppress
            "mbs_threshold": self.mbs_threshold.get(),
            "mbs_threshold_type": self.mbs_threshold_type.get(),
            "mbs_open_kernel_type": self.mbs_open_kernel_type.get(),
            "mbs_open_kernel_size": self.mbs_open_kernel_size.get(),
            "mbs_dilate_kernel_type": self.mbs_dilate_kernel_type.get(),
            "mbs_dilate_kernel_size": self.mbs_dilate_kernel_size.get(),
            "mbs_dilate_iterations": self.mbs_dilate_iterations.get(),
            "mbs_blur_kernel": self.mbs_blur_kernel.get(),
            "mbs_edge_smooth": self.mbs_edge_smooth.get(),
            "hr_canny_low": self.hr_canny_low.get(),
            "hr_canny_high": self.hr_canny_high.get(),
            "hr_hough_factor": self.hr_hough_factor.get(),
            "hr_blur_margin": self.hr_blur_margin.get(),
            "svd_k": self.svd_k.get(),
            "svd_downsample": self.svd_downsample.get(), # Added missing param
            "post_log_enable": self.post_log_enable.get(),
            "post_log_sigma": self.post_log_sigma.get(),
            "post_log_kernel": self.post_log_kernel.get(),
            "post_median_k": self.post_median_k.get(),
            "post_noise_area": self.post_noise_area.get(),
            "post_isolated_dist": self.post_isolated_dist.get(),
            "post_dilate_k": self.post_dilate_k.get(),
            "post_max_components": self.post_max_components.get(),
            "post_max_area_ratio": self.post_max_area_ratio.get(),
            "post_min_area": self.post_min_area.get(),
            "post_ar_area_threshold": self.post_ar_area_threshold.get(),
            "post_min_ar": self.post_min_ar.get(),
            "post_max_solidity": self.post_max_solidity.get(),
            "post_large_area_threshold": self.post_large_area_threshold.get(),
            "judge_min_count": self.judge_min_count.get(),
            "judge_min_ratio": self.judge_min_ratio.get()
        }

    def apply_config_dict(self, cfg):
        """Apply a dictionary of parameters to the UI variables."""
        try:
            self.step1_method.set(cfg.get("step1_method", "BlackHat"))
            # Pre-Opening
            self.pre_open_kernel_type.set(cfg.get("pre_open_kernel_type", "RECT"))
            self.pre_open_kernel_size.set(cfg.get("pre_open_kernel_size", 5))
            self.pre_open_iterations.set(cfg.get("pre_open_iterations", 1))
            self.th_kernel_size.set(cfg.get("th_kernel_size", 15))
            self.th_kernel_shape.set(cfg.get("th_kernel_shape", "RECT"))
            self.homo_gamma_l.set(cfg.get("homo_gamma_l", 0.5))
            self.homo_gamma_h.set(cfg.get("homo_gamma_h", 2.0))
            self.homo_d0.set(cfg.get("homo_d0", 30))
            self.homo_c.set(cfg.get("homo_c", 1.5))
            self.ds_scale.set(cfg.get("ds_scale", 0.25))
            self.ds_method.set(cfg.get("ds_method", "gaussian"))
            self.ds_kernel.set(cfg.get("ds_kernel", 41))
            self.ds_offset.set(cfg.get("ds_offset", 10.0))
            self.ds_norm_mode.set(cfg.get("ds_norm_mode", "divide"))
            self.ds_invert.set(cfg.get("ds_invert", False))
            # Mask-Blur-Suppress
            self.mbs_threshold.set(cfg.get("mbs_threshold", 240))
            self.mbs_threshold_type.set(cfg.get("mbs_threshold_type", "BINARY"))
            self.mbs_open_kernel_type.set(cfg.get("mbs_open_kernel_type", "RECT"))
            self.mbs_open_kernel_size.set(cfg.get("mbs_open_kernel_size", 61))
            self.mbs_dilate_kernel_type.set(cfg.get("mbs_dilate_kernel_type", "RECT"))
            self.mbs_dilate_kernel_size.set(cfg.get("mbs_dilate_kernel_size", 101))
            self.mbs_dilate_iterations.set(cfg.get("mbs_dilate_iterations", 1))
            self.mbs_blur_kernel.set(cfg.get("mbs_blur_kernel", 101))
            self.mbs_edge_smooth.set(cfg.get("mbs_edge_smooth", 51))
            self.hr_canny_low.set(cfg.get("hr_canny_low", 50))
            self.hr_canny_high.set(cfg.get("hr_canny_high", 150))
            self.hr_hough_factor.set(cfg.get("hr_hough_factor", 0.13))
            self.hr_blur_margin.set(cfg.get("hr_blur_margin", 100))
            self.svd_k.set(cfg.get("svd_k", 10))
            self.svd_downsample.set(cfg.get("svd_downsample", 0.5)) # Added missing param
            self.post_log_enable.set(cfg.get("post_log_enable", False))
            self.post_log_sigma.set(cfg.get("post_log_sigma", 2.0))
            self.post_log_kernel.set(cfg.get("post_log_kernel", 7))
            self.post_median_k.set(cfg.get("post_median_k", 5))
            self.post_noise_area.set(cfg.get("post_noise_area", 20))
            self.post_isolated_dist.set(cfg.get("post_isolated_dist", 20))
            self.post_dilate_k.set(cfg.get("post_dilate_k", 7))
            self.post_max_components.set(cfg.get("post_max_components", 50))
            self.post_max_area_ratio.set(cfg.get("post_max_area_ratio", 10.0))
            self.post_min_area.set(cfg.get("post_min_area", 20))
            self.post_ar_area_threshold.set(cfg.get("post_ar_area_threshold", 200))
            self.post_min_ar.set(cfg.get("post_min_ar", 2.0))
            self.post_max_solidity.set(cfg.get("post_max_solidity", 0.7))
            self.post_large_area_threshold.set(cfg.get("post_large_area_threshold", 100))
            self.judge_min_count.set(cfg.get("judge_min_count", 1))
            self.judge_min_ratio.set(cfg.get("judge_min_ratio", 0.01))
            print("Config applied successfully.")
        except Exception as e:
            print(f"Error applying config: {e}")
            messagebox.showerror("配置错误", f"应用配置时出错: {e}")

    def _save_config(self):
        cfg = self.get_config_dict()
        try:
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=4)
            messagebox.showinfo("配置", "默认配置已保存！")
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {e}")
        
    def _load_config(self):
        if not os.path.exists(self.config_path): return
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            self.apply_config_dict(cfg)
        except Exception as e:
            print(f"Error loading config: {e}")
            
    def export_config(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if path:
            cfg = self.get_config_dict()
            try:
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=4)
                messagebox.showinfo("导出成功", f"配置已导出至:\n{path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"无法写入文件: {e}")

    def import_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            try:
                with open(path, "r") as f:
                    cfg = json.load(f)
                self.apply_config_dict(cfg)
                messagebox.showinfo("导入成功", "配置已加载并应用。")
                self.update_display() # Refresh display with new params
            except Exception as e:
                messagebox.showerror("导入失败", f"无法读取文件: {e}")

    def _create_rejected_overlay(self):
        """Create an overlay showing both kept (red) and rejected (blue) components"""
        if self.img_step0_original is None or self.rotation_matrix is None:
            return self.img_restored_overlay
        
        h0, w0 = self.img_step0_original.shape[:2]
        
        # Scale masks if needed
        if self.img_step2_rectified is not None:
            hr, wr = self.img_step2_rectified.shape[:2]
            if self.img_step4_filtered.shape[0] != hr or self.img_step4_filtered.shape[1] != wr:
                kept_scaled = cv2.resize(self.img_step4_filtered, (wr, hr), interpolation=cv2.INTER_NEAREST)
                rejected_scaled = cv2.resize(self.img_rejected_components, (wr, hr), interpolation=cv2.INTER_NEAREST)
            else:
                kept_scaled = self.img_step4_filtered
                rejected_scaled = self.img_rejected_components
        else:
            kept_scaled = self.img_step4_filtered
            rejected_scaled = self.img_rejected_components
        
        # Inverse warp both masks
        kept_restored = cv2.warpAffine(kept_scaled, self.rotation_matrix, (w0, h0), 
                                       flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
        rejected_restored = cv2.warpAffine(rejected_scaled, self.rotation_matrix, (w0, h0), 
                                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
        
        # Create overlay: kept in red, rejected in blue
        overlay = cv2.cvtColor(self.img_step0_original, cv2.COLOR_GRAY2BGR)
        overlay[kept_restored > 127] = [0, 0, 255]     # BGR Red for kept
        overlay[rejected_restored > 127] = [255, 0, 0]  # BGR Blue for rejected
        
        return overlay

    # --- Homomorphic Filter Helper ---
    def apply_homomorphic_filter(self, img, gamma_l, gamma_h, d0, c):
        """Apply homomorphic filtering to correct uneven lighting."""
        # Log transform
        img_float = np.float32(img)
        img_log = np.log1p(img_float)
        
        # FFT
        fft = np.fft.fft2(img_log)
        fft_shift = np.fft.fftshift(fft)
        
        # Create filter mask
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(u, v, indexing='ij')
        D_sq = U**2 + V**2
        
        term = -c * (D_sq / (d0**2 + 1e-5))
        H = (gamma_h - gamma_l) * (1 - np.exp(term)) + gamma_l
        
        # Apply filter
        filtered_fft = fft_shift * H
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_fft)
        img_back_log = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back_log)
        
        # Inverse Log
        img_exp = np.expm1(img_back)
        
        # Normalize to 0-255
        img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
        result = np.uint8(np.clip(img_norm, 0, 255))
        result = 255 - result  # Invert to emphasize bright cracks after homomorphic filtering
        
        return result

    # --- Downsample Background Helpers ---
    def estimate_background_downsample(self, img, scale=0.25, method="gaussian", kernel=41):
        if img is None:
            return None
        if img.ndim != 2:
            raise ValueError("下采样背景估计仅支持灰度图")
        scale = min(max(scale, 0.01), 0.99)
        h, w = img.shape[:2]
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

        if kernel % 2 == 0:
            kernel += 1
        method = method.lower()
        if method == "gaussian":
            blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)
        elif method == "median":
            blurred = cv2.medianBlur(small, kernel)
        elif method == "opening":
            k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
            blurred = cv2.morphologyEx(small, cv2.MORPH_OPEN, k_elem)
        else:
            raise ValueError("method 必须是 gaussian/median/opening 之一")

        background = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
        return background

    def normalize_divide(self, img, background, eps=1e-3):
        img_f = img.astype(np.float32)
        bg_f = background.astype(np.float32)
        mean_val = np.mean(img_f)
        result = img_f / (bg_f + eps) * mean_val
        return np.clip(result, 0, 255).astype(np.uint8)

    def normalize_subtract(self, img, background, offset=10.0):
        img_f = img.astype(np.float32)
        bg_f = background.astype(np.float32)
        result = img_f - bg_f + offset
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- Mask-Blur-Suppress Helper ---
    def _get_kernel(self, kernel_type: str, size: int):
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1
        kmap = {
            "RECT": cv2.MORPH_RECT,
            "ELLIPSE": cv2.MORPH_ELLIPSE,
            "CROSS": cv2.MORPH_CROSS
        }
        return cv2.getStructuringElement(kmap.get(kernel_type, cv2.MORPH_RECT), (size, size))

    def apply_mask_blur_suppress(
        self,
        img_gray: np.ndarray,
        threshold: int = 240,
        threshold_type: str = "BINARY",
        open_kernel_type: str = "RECT",
        open_kernel_size: int = 61,
        dilate_kernel_type: str = "RECT",
        dilate_kernel_size: int = 101,
        dilate_iterations: int = 1,
        blur_kernel: int = 101,
        edge_smooth: int = 51,
    ) -> np.ndarray:
        """在掩模区域进行高斯模糊并用平滑掩模混合，适配灰度图。"""
        if img_gray is None:
            return None
        src = img_gray
        # 1) 二值化
        th_type = cv2.THRESH_BINARY if threshold_type == "BINARY" else cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(src, int(threshold), 255, th_type)

        # 2) 开运算
        open_kernel = self._get_kernel(open_kernel_type, int(open_kernel_size))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

        # 3) 膨胀
        dilate_kernel = self._get_kernel(dilate_kernel_type, int(dilate_kernel_size))
        dilated = cv2.dilate(opened, dilate_kernel, iterations=max(1, int(dilate_iterations)))

        # 4) 整体模糊
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(src.astype(np.float32), (int(blur_kernel), int(blur_kernel)), 0)

        # 5) 平滑掩模
        if edge_smooth % 2 == 0:
            edge_smooth += 1
        mask_f = (dilated.astype(np.float32) / 255.0)
        smooth = cv2.GaussianBlur(mask_f, (int(edge_smooth), int(edge_smooth)), 0)
        smooth = cv2.GaussianBlur(smooth, (int(edge_smooth), int(edge_smooth)), 0)
        # gamma 调整使过渡更自然
        smooth = np.power(np.clip(smooth, 0.0, 1.0), 0.7)

        # 6) 混合
        src_f = src.astype(np.float32)
        result = src_f * (1.0 - smooth) + blurred * smooth
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- New Step 4 Pipeline Methods ---
    # 已移除：低灰度阈值处理（Step 4.0）
    
    def apply_step4_log(self):
        """Step 4.0.5: Apply Laplacian of Gaussian (LoG) filter"""
        if self.img_step3_svd is None:
            messagebox.showwarning("错误", "请先完成 SVD 分解 (步骤 3)")
            return
        
        # 如果已经有step4_current，就在它基础上处理，否则从SVD结果开始
        if self.img_step4_current is not None:
            img = self.img_step4_current.copy()
        else:
            img = self.img_step3_svd.copy()
        
        sigma = self.post_log_sigma.get()
        kernel_size = self.post_log_kernel.get()
        
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            # 先做高斯模糊
            gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            
            # 应用Laplacian算子
            laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
            
            # 转换回uint8，使用绝对值
            laplacian = np.abs(laplacian)
            laplacian = np.uint8(np.clip(laplacian, 0, 255))
            
            self.img_step4_current = laplacian
            self.lbl_status.config(text=f"4.0.5: LoG 滤波 (sigma={sigma}, kernel={kernel_size}) 完成")
            self.update_display(process_current=False)
        except Exception as e:
            messagebox.showerror("错误", f"LoG 滤波失败: {str(e)}")
    
    def apply_step4_median(self):
        """Step 4.1: Apply median blur to SVD result"""
        if self.img_step3_svd is None:
            messagebox.showwarning("错误", "请先完成 SVD 分解 (步骤 3)")
            return
        
        # 如果已经有step4_current，就在它基础上处理，否则从SVD结果开始
        if self.img_step4_current is not None:
            img = self.img_step4_current.copy()
        else:
            img = self.img_step3_svd.copy()
        
        k = self.post_median_k.get()
        if k % 2 == 0:  # Ensure odd kernel size
            k += 1
        
        self.img_step4_current = cv2.medianBlur(img, k)
        self.lbl_status.config(text=f"4.1: 中值滤波 (核={k}) 完成")
        self.update_display(process_current=False)
    
    def reset_step4(self):
        """重置步骤4到初始状态（SVD输出）"""
        self.img_step4_current = None
        self.img_step4_filtered = None
        self.img_restored_overlay = None
        self.step4_all_executed = False
        
        # Reset judgment
        self.judge_result_text.set("未知")
        self.judge_result_color = "gray"
        if hasattr(self, 'lbl_judge'):
            self.lbl_judge.config(foreground=self.judge_result_color)
        
        self.lbl_status.config(text="步骤4已重置，回到 SVD 输出状态")
        self.update_display(process_current=False)
    
    def apply_step4_binary(self):
        """Step 4.2: Apply Adaptive Automatic Thresholding"""
        if self.img_step4_current is None or len(self.img_step4_current.shape) == 3:
            messagebox.showwarning("错误", "请先执行中值滤波")
            return
        
        max_comp = self.post_max_components.get()
        max_ratio = self.post_max_area_ratio.get() / 100.0
        
        img_h, img_w = self.img_step4_current.shape[:2]
        total_pixels = img_h * img_w
        
        # Search from low threshold up
        best_threshold = 0
        best_binary = None
        found = False
        
        # Use a copy to avoid modifying the current image during search if we were using it in place (though here we use threshold)
        src_img = self.img_step4_current
        
        for threshold in range(0, 256):
            _, binary = cv2.threshold(src_img, threshold, 255, cv2.THRESH_BINARY)
            
            # Count connected components
            num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
            num_components = num_labels - 1  # Exclude background
            
            # Calculate foreground area ratio
            foreground_pixels = np.count_nonzero(binary)
            area_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0
            
            # Check conditions
            if num_components <= max_comp and area_ratio <= max_ratio:
                best_threshold = threshold
                best_binary = binary.copy()
                found = True
                break
        
        if found:
            self.img_step4_current = best_binary
            num_labels, _ = cv2.connectedComponents(best_binary, connectivity=8)
            foreground = np.count_nonzero(best_binary)
            ratio = (foreground / total_pixels) * 100
            self.lbl_status.config(text=f"4.2: 自适应二值化成功 (阈值={best_threshold}, 分量={num_labels-1}, 占比={ratio:.2f}%)")
        else:
            # Fallback or warning
            self.lbl_status.config(text="4.2: 自适应二值化失败 (未找到满足条件的阈值)")
            messagebox.showwarning("警告", f"未找到满足条件的阈值\n(最大分量:{max_comp}, 最大占比:{max_ratio*100:.1f}%)")
            
        self.update_display(process_current=False)
    
    def apply_step4_remove_small(self):
        """Step 4.3: Remove small connected components"""
        if self.img_step4_current is None or len(self.img_step4_current.shape) == 3:
            messagebox.showwarning("错误", "请先执行二值化")
            return
        
        max_area = self.post_noise_area.get()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(self.img_step4_current, connectivity=8)
        
        # Create result image
        result = np.zeros_like(self.img_step4_current)
        removed_count = 0
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                result[labels == i] = 255
            else:
                removed_count += 1
        
        self.img_step4_current = result
        self.lbl_status.config(text=f"4.3: 删除 {removed_count} 个小面积组件")
        self.update_display(process_current=False)
    
    def apply_step4_remove_isolated(self):
        """Step 4.4: Remove isolated components"""
        if self.img_step4_current is None:
            messagebox.showwarning("错误", "请先执行前序步骤")
            return
        
        dist = self.post_isolated_dist.get()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.img_step4_current, connectivity=8)
        if num_labels <= 1:
            self.update_display(process_current=False)
            return

        k_size = dist // 2
        if k_size < 1: k_size = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k_size+1, 2*k_size+1))
        dilated = cv2.dilate(self.img_step4_current, kernel)
        num_dilated, labels_dilated = cv2.connectedComponents(dilated, connectivity=8)
        
        # Optimized mapping using unique pairs
        mask = self.img_step4_current > 0
        pair_labels = np.column_stack((labels[mask], labels_dilated[mask]))
        unique_pairs = np.unique(pair_labels, axis=0)
        
        ol_list = unique_pairs[:, 0]
        dl_list = unique_pairs[:, 1]
        
        # Count how many original labels per dilated label
        dl_counts = np.bincount(dl_list)
        isolated_dl = np.where(dl_counts == 1)[0]
        
        # Identification of isolated original labels
        is_isolated_ol = np.isin(dl_list, isolated_dl)
        isolated_ol_labels = ol_list[is_isolated_ol]
        
        # Optimized removal using boolean indexing
        result = self.img_step4_current.copy()
        remove_mask = np.isin(labels, isolated_ol_labels)
        result[remove_mask] = 0
            
        self.img_step4_current = result
        self.lbl_status.config(text=f"4.4: 删除 {len(isolated_ol_labels)} 个孤立组件")
        self.update_display(process_current=False)
    
    def apply_step4_dilate(self):
        """Step 4.5: Apply dilation"""
        if self.img_step4_current is None:
            messagebox.showwarning("错误", "请先执行前序步骤")
            return
        
        k = self.post_dilate_k.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.img_step4_current = cv2.dilate(self.img_step4_current, kernel, iterations=1)
        self.lbl_status.config(text=f"4.5: 膨胀 (核={k}) 完成")
        self.update_display(process_current=False)

    def _skeletonize(self, binary_mask: np.ndarray):
        if binary_mask is None:
            return None
        bin_img = (binary_mask > 0).astype(np.uint8)
        # Prefer ximgproc.thinning if available
        try:
            from cv2 import ximgproc
            return ximgproc.thinning(bin_img)
        except Exception:
            pass
        # Morphological skeleton fallback
        skel = np.zeros_like(bin_img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        eroded = bin_img.copy()
        while True:
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(eroded, opened)
            skel = cv2.bitwise_or(skel, temp)
            eroded = cv2.erode(eroded, element)
            if cv2.countNonZero(eroded) == 0:
                break
        return (skel > 0).astype(np.uint8) * 255
    
    def apply_step4_filter(self):
        """Step 4.6: Apply final blob filtering"""
        if self.img_step4_current is None:
            messagebox.showwarning("错误", "请先执行前序步骤")
            return
        
        cnts = cv2.findContours(self.img_step4_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        min_area = self.post_min_area.get()
        ar_area_threshold = self.post_ar_area_threshold.get()
        min_ar = self.post_min_ar.get()
        max_sol = self.post_max_solidity.get()
        
        # Create final mask and rejected mask
        final_mask = np.zeros_like(self.img_step4_current)
        rejected_mask = np.zeros_like(self.img_step4_current)
        kept_count = 0
        
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area: 
                continue
            
            # Aspect Ratio via minAreaRect
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (width, height), angle = rect
            major = max(width, height)
            minor = min(width, height)
            ar = major / (minor if minor > 0 else 0.01)
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            meets_ar = ar >= min_ar
            meets_area_override = area >= ar_area_threshold
            standard_pass = (meets_ar or meets_area_override) and solidity <= max_sol
            
            if standard_pass:
                cv2.drawContours(final_mask, [cnt], -1, 255, -1)
                kept_count += 1
            else:
                cv2.drawContours(rejected_mask, [cnt], -1, 255, -1)
        
        self.img_step4_filtered = final_mask
        self.img_rejected_components = rejected_mask
        
        # --- Judgment Logic ---
        total_pixels = final_mask.size
        crack_pixels = cv2.countNonZero(final_mask)
        ratio = (crack_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Count distinct cracks in final mask
        final_cnts = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        count = len(final_cnts)
        
        jud_count = self.judge_min_count.get()
        jud_ratio = self.judge_min_ratio.get()
        
        # Persist metrics for batch/reporting
        self.judge_count = count
        self.judge_ratio = ratio
        self.judge_defect = count >= jud_count and ratio >= jud_ratio

        if self.judge_defect:
            self.judge_result_text.set(f"FAIL (检测到裂纹)\n数量:{count}, 占比:{ratio:.3f}%")
            self.judge_result_color = "red"
        else:
            self.judge_result_text.set(f"PASS (合格)\n数量:{count}, 占比:{ratio:.3f}%")
            self.judge_result_color = "green"
        
        if hasattr(self, 'lbl_judge'):
            self.lbl_judge.config(foreground=self.judge_result_color)

        # --- Generate Restored Overlays ---
        self.img_restored_overlay = None
        self.img_restored_cc_overlay = None
        self.img_restored_skeleton_overlay = None
        if self.img_step0_original is not None and self.rotation_matrix is not None:
            h0, w0 = self.img_step0_original.shape[:2]
            
            # If Step 3 was downsampled, scale mask back to rectified size
            if self.img_step2_rectified is not None:
                hr, wr = self.img_step2_rectified.shape[:2]
                if final_mask.shape[0] != hr or final_mask.shape[1] != wr:
                    mask_scaled = cv2.resize(final_mask, (wr, hr), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_scaled = final_mask
            else:
                mask_scaled = final_mask

            # Inverse Warp the mask back to original size
            restored_mask = cv2.warpAffine(mask_scaled, self.rotation_matrix, (w0, h0), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
            
            # 1) 单色掩膜叠加（红色）
            original_color = cv2.cvtColor(self.img_step0_original, cv2.COLOR_GRAY2BGR)
            overlay_mask = restored_mask > 127
            original_color[overlay_mask] = [0, 0, 255]  # BGR Red
            self.img_restored_overlay = original_color

            # 2) 连通域彩色覆盖
            num_cc, cc_labels = cv2.connectedComponents(mask_scaled, connectivity=8)
            if num_cc > 1:
                hsv = np.zeros((mask_scaled.shape[0], mask_scaled.shape[1], 3), dtype=np.uint8)
                for lbl in range(1, num_cc):
                    color = ((lbl * 37) % 180, 200, 255)  # generate distinct HSV hues
                    hsv[cc_labels == lbl] = color
                cc_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cc_restored = cv2.warpAffine(cc_bgr, self.rotation_matrix, (w0, h0), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
                base = cv2.cvtColor(self.img_step0_original, cv2.COLOR_GRAY2BGR)
                mask_overlay = cv2.warpAffine(mask_scaled, self.rotation_matrix, (w0, h0), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
                base[mask_overlay > 0] = cc_restored[mask_overlay > 0]
                self.img_restored_cc_overlay = base

            # 3) 骨架拟合覆盖
            skeleton = self._skeletonize(mask_scaled)
            if skeleton is not None:
                sk_restored = cv2.warpAffine(skeleton, self.rotation_matrix, (w0, h0), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
                base = cv2.cvtColor(self.img_step0_original, cv2.COLOR_GRAY2BGR)
                base[sk_restored > 0] = [0, 0, 255]  # red
                self.img_restored_skeleton_overlay = base
        
        self.lbl_status.config(text=f"4.6: 筛选完成，保留 {kept_count} 个裂纹组件")
        self.update_display(process_current=False)
    
    def execute_step4_all(self, auto_trigger=False):
        """Execute the entire Step 4 pipeline sequentially"""
        if self.img_step3_svd is None:
            messagebox.showwarning("错误", "请先完成 SVD 分解 (步骤 3)")
            return
        
        # Reset to start from SVD
        self.img_step4_current = None
        self.step4_all_executed = False
        
        self.lbl_status.config(text="执行完整流程中...")
        self.root.update_idletasks()
        
        # Run all steps in sequence
        try:
            # 可选：LoG滤波作为起始步骤
            if self.post_log_enable.get():
                self.apply_step4_log()
            self.apply_step4_median()
            self.apply_step4_binary()
            self.apply_step4_remove_small()
            self.apply_step4_remove_isolated()
            self.apply_step4_dilate()
            self.apply_step4_filter()
            
            self.lbl_status.config(text="✓ 完整流程执行完成！")
            if not auto_trigger:
                messagebox.showinfo("完成", "Step 4 完整流程执行成功！")
            self.step4_all_executed = True
        except Exception as e:
            self.lbl_status.config(text=f"错误: {str(e)}")
            messagebox.showerror("错误", f"执行流程时出错:\n{str(e)}")

    def process_single_image(self, img_path, output_root="final_results", quiet=True, save_keys=None, flatten_if_single=True):
        """Run the full pipeline on a single image path (headless/batch)."""
        prev_headless = self.headless_mode
        self.headless_mode = True
        self.img_path = img_path
        self.step4_all_executed = False
        # Load gray and scale to 0.25 like GUI
        self.img_step0_original = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if self.img_step0_original is None:
            raise RuntimeError(f"无法读取图像: {img_path}")
        h, w = self.img_step0_original.shape[:2]
        nh, nw = max(1, int(h * 0.25)), max(1, int(w * 0.25))
        self.img_step0_original = cv2.resize(self.img_step0_original, (nw, nh), interpolation=cv2.INTER_AREA)

        # Step1-3
        self.process_step(1)
        self.process_step(2)
        self.process_step(3)

        # Step4 full pipeline
        self.execute_step4_all(auto_trigger=True)

        # Save results
        base = os.path.splitext(os.path.basename(img_path))[0]
        flatten = bool(save_keys) and len(save_keys) == 1 and flatten_if_single
        result = self.save_all_results(out_root=output_root, base_name=base, quiet=quiet, save_keys=save_keys, flatten=flatten)
        self.headless_mode = prev_headless
        return result



if __name__ == "__main__":
    root = tk.Tk()
    # High DPI aware
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    
    app = IntegratedCrackGUI(root)
    root.mainloop()
