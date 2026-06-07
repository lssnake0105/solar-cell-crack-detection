import os
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class SegAnnotator:
    def __init__(self, root, dataset_root, subset="train"):
        self.root = root
        self.root.title("SVD 裂纹分割标注 GUI（线段模式 + 连通域擦除）")
        self.dataset_root = dataset_root
        self.subset = subset
        self.images_dir = os.path.join(dataset_root, "images", subset)
        self.masks_dir = os.path.join(dataset_root, "masks", subset)
        os.makedirs(self.masks_dir, exist_ok=True)

        self.files = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.lower().endswith((".jpg", ".png"))]
        self.idx = 0

        self.canvas = tk.Canvas(root, width=960, height=720, bg="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        ctrl = ttk.Frame(root)
        ctrl.pack(fill=tk.X)
        ttk.Button(ctrl, text="上一张", command=self.prev).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="下一张", command=self.next).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="提交线段", command=self.commit_polyline).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="撤销点", command=self.undo_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="清空掩膜", command=self.clear_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="保存掩膜", command=self.save_mask).pack(side=tk.LEFT, padx=5)
        ttk.Label(ctrl, text="线宽:").pack(side=tk.LEFT, padx=5)
        self.thick_var = tk.IntVar(value=3)
        self.thick_scale = ttk.Scale(ctrl, from_=1, to=20, orient=tk.HORIZONTAL, command=self.on_thick_change)
        self.thick_scale.set(self.thick_var.get())
        self.thick_scale.pack(side=tk.LEFT, padx=5)
        self.erase_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="擦除连通域", variable=self.erase_var).pack(side=tk.LEFT, padx=10)

        self.img_cv = None
        self.img_tk = None
        self.scale = 1.0
        self.disp_size = (960, 720)
        self.mask = None        # 原始大小的掩膜 (H,W) uint8
        self.points = []        # 当前线段的点（画布坐标）

        self.load_current()

        # --- Keyboard Shortcuts ---
        # Enter: 提交当前线段
        self.root.bind('<Return>', lambda e: self.commit_polyline())
        # s/S: 保存掩膜
        self.root.bind('<s>', lambda e: self.save_mask())
        self.root.bind('<S>', lambda e: self.save_mask())
        # 方向键：切换图片（左/上=上一张，右/下=下一张）
        self.root.bind('<Left>', lambda e: self.prev())
        self.root.bind('<Up>', lambda e: self.prev())
        self.root.bind('<Right>', lambda e: self.next())
        self.root.bind('<Down>', lambda e: self.next())

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_current()

    def next(self):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.load_current()

    def load_current(self):
        if not self.files:
            return
        path = self.files[self.idx]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.img_cv = img
        h, w = img.shape[:2]
        # 载入同名掩膜（如存在）
        base = os.path.splitext(os.path.basename(path))[0]
        mpath = os.path.join(self.masks_dir, base + ".png")
        m = None
        if os.path.exists(mpath):
            m = cv2.imdecode(np.fromfile(mpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if m is None:
            m = np.zeros((h, w), dtype=np.uint8)
        self.mask = m

        cw = self.canvas.winfo_width() or self.disp_size[0]
        ch = self.canvas.winfo_height() or self.disp_size[1]
        self.scale = min(cw / max(1, w), ch / max(1, h)) * 0.95
        self.update_display()

    def on_thick_change(self, val):
        try:
            self.thick_var.set(int(float(val)))
        except Exception:
            pass

    def _canvas_to_img(self, x, y):
        h, w = self.img_cv.shape[:2]
        cw = self.canvas.winfo_width() or self.disp_size[0]
        ch = self.canvas.winfo_height() or self.disp_size[1]
        nw = int(w * self.scale)
        nh = int(h * self.scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        rx = int((x - ox) / self.scale)
        ry = int((y - oy) / self.scale)
        return rx, ry

    def on_click(self, event):
        # 如果是擦除连通域模式，则根据点击点所在的连通域进行擦除
        if self.erase_var.get():
            rx, ry = self._canvas_to_img(event.x, event.y)
            h, w = self.mask.shape[:2]
            if 0 <= rx < w and 0 <= ry < h:
                if self.mask[ry, rx] > 0:
                    lbls = cv2.connectedComponents((self.mask > 0).astype(np.uint8))[1]
                    lab = int(lbls[ry, rx])
                    self.mask[lbls == lab] = 0
                    self.update_display()
            return
        # 否则为线段模式：点击添加点，并在画布上显示临时线段
        self.points.append((event.x, event.y))
        n = len(self.points)
        if n >= 2:
            x0, y0 = self.points[-2]
            x1, y1 = self.points[-1]
            self.canvas.create_line(x0, y0, x1, y1, fill="lime", width=max(1, self.thick_var.get()))
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, outline="yellow", fill="yellow")

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
            self.update_display()

    def update_display(self):
        h, w = self.img_cv.shape[:2]
        cw = self.canvas.winfo_width() or self.disp_size[0]
        ch = self.canvas.winfo_height() or self.disp_size[1]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        disp = cv2.resize(self.img_cv, (nw, nh))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        # 叠加掩膜
        m_small = cv2.resize(self.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(disp)
        overlay[..., 0] = 255  # 红色通道（RGB）注意：这里用红色=裂纹
        alpha = (m_small.astype(np.float32)/255.0) * 0.5
        disp = (disp.astype(np.float32)*(1-alpha)[...,None] + overlay.astype(np.float32)*alpha[...,None]).astype(np.uint8)
        self.canvas.delete("all")
        pil = Image.fromarray(disp)
        self.img_tk = ImageTk.PhotoImage(pil)
        self.canvas.create_image(cw // 2, ch // 2, image=self.img_tk)
        self.canvas.image = self.img_tk
        self.canvas.create_text(10, 10, anchor="nw", fill="white", text=f"{os.path.basename(self.files[self.idx])} ({self.idx+1}/{len(self.files)})  线宽:{self.thick_var.get()}  擦除连通域:{'是' if self.erase_var.get() else '否'}")
        # 重新绘制临时线段与点
        if len(self.points) >= 1:
            for i in range(1, len(self.points)):
                x0, y0 = self.points[i-1]
                x1, y1 = self.points[i]
                self.canvas.create_line(x0, y0, x1, y1, fill="lime", width=max(1, self.thick_var.get()))
            for (x, y) in self.points:
                self.canvas.create_oval(x-2, y-2, x+2, y+2, outline="yellow", fill="yellow")

    def clear_polygons(self):
        self.points.clear()

    def save_mask(self):
        if self.img_cv is None or self.mask is None:
            return
        # 保存到 masks/<subset>/<basename>.png
        img_path = self.files[self.idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(self.masks_dir, base + ".png")
        ok, buf = cv2.imencode('.png', self.mask)
        if not ok:
            messagebox.showerror("错误", "掩膜编码失败")
            return
        buf.tofile(out_path)
        messagebox.showinfo("成功", f"已保存掩膜:\n{out_path}")

    def commit_polyline(self):
        # 将当前 points 形成的线段绘制到掩膜上（原图坐标），线宽可调
        if len(self.points) < 2:
            messagebox.showwarning("提示", "请至少点击两个点形成线段")
            return
        h, w = self.mask.shape[:2]
        thick = max(1, int(self.thick_var.get()))
        # 转换到原图坐标并逐段绘制
        pts_img = []
        for (x, y) in self.points:
            rx, ry = self._canvas_to_img(x, y)
            rx = max(0, min(w-1, rx))
            ry = max(0, min(h-1, ry))
            pts_img.append((rx, ry))
        for i in range(1, len(pts_img)):
            x0, y0 = pts_img[i-1]
            x1, y1 = pts_img[i]
            cv2.line(self.mask, (x0, y0), (x1, y1), 255, thick, lineType=cv2.LINE_AA)
        self.points.clear()
        self.update_display()

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.update_display()


def main():
    parser = argparse.ArgumentParser(description="Polygon mask annotator for SVD segmentation")
    parser.add_argument("--root", required=True, help="dataset root")
    parser.add_argument("--subset", default="train", help="train or val")
    args = parser.parse_args()

    root = tk.Tk()
    app = SegAnnotator(root, dataset_root=args.root, subset=args.subset)
    root.mainloop()


if __name__ == "__main__":
    main()
