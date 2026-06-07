import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import datetime
import os
import sys
from pathlib import Path

# Allow importing the existing processing logic
sys.path.append(str(Path(__file__).parent))
import integrated_crack_detector_gui as gui
import generate_batch_report as report

DEFAULT_OUT = Path(__file__).parent / "batch_result"

SAVE_OPTIONS = [
    ("原始图", "0_original"),
    ("预处理", "1_blackhat"),
    ("纠偏", "2_rectified"),
    ("SVD", "3_svd"),
    ("二值/形态", "4_binary"),
    ("最终筛选二值", "4_filtered"),
    ("最终叠加", "5_overlay"),
    ("连通域叠加", "5_cc"),
    ("骨架叠加", "5_skeleton"),
]

DEFAULT_SELECTED = {"5_skeleton"}


def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p


class BatchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("批处理裂纹检测")
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(DEFAULT_OUT))
        self.config_path = tk.StringVar(value="crack_detection_config.json")
        self.running = False
        self.queue = queue.Queue()
        self.selected_keys = {k for k in DEFAULT_SELECTED}
        self.gen_report = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(frm); row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="输入文件夹:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row1, text="选择...", command=self.choose_input).pack(side=tk.LEFT)

        row2 = ttk.Frame(frm); row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="输出文件夹:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.output_dir, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row2, text="选择...", command=self.choose_output).pack(side=tk.LEFT)

        row3 = ttk.Frame(frm); row3.pack(fill=tk.X, pady=5)
        ttk.Label(row3, text="配置文件:").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.config_path, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(row3, text="选择...", command=self.choose_config).pack(side=tk.LEFT)

        # Save options
        lf = ttk.LabelFrame(frm, text="保存哪些步骤 (默认仅结果图)", padding=10)
        lf.pack(fill=tk.X, pady=10)
        self.check_vars = {}
        for text, key in SAVE_OPTIONS:
            var = tk.BooleanVar(value=(key in self.selected_keys))
            self.check_vars[key] = var
            ttk.Checkbutton(lf, text=text, variable=var, command=self._update_selected).pack(anchor=tk.W)

        ttk.Checkbutton(frm, text="生成检测报告 (HTML)", variable=self.gen_report).pack(anchor=tk.W, pady=(0,10))

        # Run button and log
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="开始批处理", command=self.start).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="停止", command=self.stop).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(frm, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)

        self.log = tk.Text(frm, height=12, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _update_selected(self):
        self.selected_keys = {k for k, v in self.check_vars.items() if v.get()}

    def choose_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)

    def choose_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def choose_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.config_path.set(path)

    def start(self):
        if self.running:
            return
        inp = Path(self.input_dir.get())
        if not inp.exists() or not inp.is_dir():
            messagebox.showwarning("提示", "请选择有效的输入文件夹")
            return
        out_root = Path(self.output_dir.get()) if self.output_dir.get() else DEFAULT_OUT
        out_root.mkdir(parents=True, exist_ok=True)
        # timestamp subfolder with input-folder prefix
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = inp.name if inp.name else "input"
        out_root = out_root / f"{prefix}_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)

        cfg_path = Path(self.config_path.get()) if self.config_path.get() else Path("crack_detection_config.json")

        self.running = True
        self.progress.start(10)
        self.log_msg(f"开始批处理，输出目录: {out_root}")

        t = threading.Thread(target=self._worker, args=(inp, out_root, cfg_path))
        t.daemon = True
        t.start()
        self.root.after(100, self._poll_queue)

    def stop(self):
        self.running = False
        self.log_msg("停止请求已发出")

    def _poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.log_msg(msg)
        except queue.Empty:
            pass
        if self.running:
            self.root.after(200, self._poll_queue)
        else:
            self.progress.stop()

    def _worker(self, inp: Path, out_root: Path, cfg_path: Path):
        # headless tk root
        root = tk.Tk()
        root.withdraw()
        app = gui.IntegratedCrackGUI(root)
        if cfg_path.exists():
            app.config_path = str(cfg_path)
            app._load_config()
        save_keys = list(self.selected_keys) if self.selected_keys else None
        processed = 0
        for img_path in iter_images(inp):
            if not self.running:
                break
            try:
                out_dir = app.process_single_image(
                    str(img_path),
                    output_root=str(out_root),
                    quiet=True,
                    save_keys=save_keys,
                    flatten_if_single=True,
                )
                processed += 1
                self.queue.put(f"[OK] {img_path} -> {out_dir}")
            except Exception as e:
                self.queue.put(f"[FAIL] {img_path}: {e}")
        # 生成报告
        if self.gen_report.get():
            try:
                out_report = report.generate_report(out_root, cfg_path, out_root / "report.html", f"裂纹检测报告 - {out_root.name}")
                self.queue.put(f"[REPORT] 报告已生成: {out_report}")
            except Exception as e:
                self.queue.put(f"[REPORT-FAIL] 生成报告失败: {e}")

        self.queue.put(f"处理完成，成功 {processed} 张，输出目录: {out_root}")
        self.running = False

    def log_msg(self, msg):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = BatchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
