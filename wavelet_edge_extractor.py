import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import tkinter.font as tkFont
import pywt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import sys
from skimage.morphology import skeletonize



class WaveletEdgeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=12)  # Increase to desired size
        self.option_add("*Font", default_font)
        self.title("Wavelet Edge Detector")
        self.geometry("1500x700")
        self.max_width = 750
        self.max_height = 450

        # --- State variables ---
        self.wavelet = tk.StringVar(value="db1")
        self.thresholds = []
        self.wavelets = []
        self.coeffs = []
        self.coherences = []
        self.crop_flags = []
        self.avg = None
        self.binary_threshold = tk.DoubleVar(value=1.0)

        # --- Main container ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # --- Top image row ---
        self.image_row = ttk.Frame(self.main_frame)
        self.image_row.pack(side="top", fill="x")

        self.orig_frame = ttk.LabelFrame(self.image_row, text="Original")
        self.orig_frame.pack(side="left", padx=5, pady=5)
        self.orig_canvas = tk.Canvas(self.orig_frame, bg="black")
        self.orig_canvas.pack()

        self.edge_frame = ttk.LabelFrame(self.image_row, text="Edges")
        self.edge_frame.pack(side="left", padx=5, pady=5)
        self.edge_canvas = tk.Canvas(self.edge_frame, bg="black")
        self.edge_canvas.pack()

        # --- Bottom controls panel ---
        ctrl_panel = ttk.Frame(self.main_frame)
        ctrl_panel.pack(side="top", fill="x", padx=10, pady=10)

        # Top row: Wavelet entry and +/– buttons
        top_ctrl_row = ttk.Frame(ctrl_panel)
        top_ctrl_row.pack(side="top", fill="x")

        ttk.Label(top_ctrl_row, text="Wavelet:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 5))
        wavelet_entry = ttk.Entry(top_ctrl_row, textvariable=self.wavelet, width=10, font=("Arial", 10))
        wavelet_entry.pack(side="left", padx=(0, 15))
        wavelet_entry.bind("<Return>", lambda e: self._on_wavelet_change())

        ttk.Button(top_ctrl_row, text="+", width=3, command=self._add_slider).pack(side="left", padx=4)
        ttk.Button(top_ctrl_row, text="–", width=3, command=self._remove_slider).pack(side="left", padx=4)

        # Sliders frame (stacking downwards)
        self.sliders_frame = ttk.Frame(ctrl_panel)
        self.sliders_frame.pack(side="top", fill="x", pady=5)

        # Load default image
        self._load_image(sys.argv[1])
        self._draw_sliders()
        self._recompute_and_update()


    def _load_image(self, path):
        max_width, max_height = 750, 700  # space allocated for one image
        # Load and convert to grayscale
        img = Image.open(path).convert("L")
        self.original_array = np.array(img, dtype=np.float32)/255.0
        # Resize proportionally to fit within (max_width, max_height)
        img.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
        self.original = img

        self.orig_canvas.config(width=img.width, height=img.height)
        self.edge_canvas.config(width=img.width, height=img.height)

        self.tk_orig = ImageTk.PhotoImage(img)
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(0, 0, anchor="nw", image=self.tk_orig)

        self.avg = self.original_array


    def _draw_sliders(self):
        """Draw all threshold sliders in self.sliders_frame."""
        for widget in self.sliders_frame.winfo_children():
            widget.destroy()

        frm = ttk.Frame(self.sliders_frame)
        frm.pack(fill="x", pady=6)
        ttk.Label(frm, text=f"Binary thresh:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 5))
        val_lbl = ttk.Label(frm, text=f"{self.binary_threshold.get():.2f}", width=5, font=("Arial", 10))
        val_lbl.pack(side="left", padx=(0, 5))
        ttk.Scale(frm, from_=0.0, to=4, variable=self.binary_threshold, orient="horizontal",
            command=lambda v, i=-1, lbl=val_lbl, vref=self.binary_threshold: (lbl.config(text=f"{vref.get():.2f}"), self._on_threshold_change(i))
        ).pack(side="left", fill="x", expand=True)


        for idx, var in enumerate(self.thresholds):
            frm = ttk.Frame(self.sliders_frame)
            frm.pack(fill="x", pady=4)  # extra padding for neat spacing
            # Threshold label (e.g., "L0 thresh:")
            ttk.Label(frm, text=f"L{idx}[{self.wavelets[idx]}] weight:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 5))
            # Value label (e.g., "1.00")
            val_lbl = ttk.Label(frm, text=f"{var.get():.2f}", width=5, font=("Arial", 10))
            val_lbl.pack(side="left", padx=(0, 5))
            # Slider
            ttk.Scale(frm, from_=0.0, to=1.0, variable=var, orient="horizontal",
                command=lambda v, i=idx, lbl=val_lbl, vref=var: (lbl.config(text=f"{vref.get():.2f}"), self._on_threshold_change(i))
            ).pack(side="left", fill="x", expand=True)


    def _compute_coherence(self, cH, cV):
        M00 = np.square(cH)
        M01 = cH*cV
        M11 = np.square(cV)

        # window = np.ones((3, 3), dtype=coherence.dtype)
        # M00 = convolve2d(M00, window, mode='same', boundary='symm')
        # M01 = convolve2d(M01, window, mode='same', boundary='symm')
        # M11 = convolve2d(M11, window, mode='same', boundary='symm')
        M00 = gaussian_filter(M00, sigma=1.0)
        M01 = gaussian_filter(M01, sigma=1.0)
        M11 = gaussian_filter(M11, sigma=1.0)

        coherence = ((M00-M11)**2+(2*M01)**2)/(M00+M11+1e-8)**2
        return coherence

    def _add_slider(self):
        wavelet = self.wavelet.get()
        if pywt.dwtn_max_level(self.avg.shape, wavelet=wavelet)==0:
            return

        self.thresholds.append(tk.DoubleVar(value=1.0))
        self.wavelets.append(wavelet)
        self._draw_sliders()

        self.crop_flags.append((self.avg.shape[0]%2==1, self.avg.shape[1]%2==1))
        self.avg, coeff = pywt.dwt2(self.avg, wavelet=wavelet)
        self.coeffs.append(coeff)
        self.coherences.append(self._compute_coherence(coeff[0], coeff[1]))

        self._recompute_and_update()

    def _remove_slider(self):
        if len(self.thresholds)==0:
            return

        self.thresholds.pop()
        wavelet = self.wavelets.pop()
        self._draw_sliders()

        self.avg = pywt.idwt2([self.avg, self.coeffs.pop()], wavelet=wavelet)
        crop_flag = self.crop_flags.pop()
        if crop_flag[0]:    self.avg = self.avg[:-1]
        if crop_flag[1]:    self.avg = self.avg[:, :-1]
        self.coherences.pop()

        self._recompute_and_update()
    

    def _on_wavelet_change(self):
        pass
    
    def _on_threshold_change(self, idx):
        self._recompute_and_update()
    
    def _recompute_and_update(self):
        edge_img = np.zeros_like(self.avg)
        # main_mask = np.ones_like(self.avg, dtype=bool)
        for i in range(len(self.thresholds)-1, -1, -1):
            cH, cV, cD = self.coeffs[i]
            mask = (self.coherences[i]*self.thresholds[i].get())
            # main_mask = np.logical_or(main_mask, mask)
            # main_mask = np.kron(main_mask, np.ones((2, 2), dtype=bool))
            edge_img = pywt.idwt2([edge_img, (cH*mask, cV*mask, cD*mask)], wavelet=self.wavelets[i])
            crop_flag = self.crop_flags[i]
            if crop_flag[0]:    edge_img = edge_img[:-1]
            if crop_flag[1]:    edge_img = edge_img[:, :-1]

            # crop_flag = self.crop_flags[i]
            # if crop_flag[0]:    main_mask = main_mask[:-1]
            # if crop_flag[1]:    main_mask = main_mask[:, :-1]

        # if len(self.thresholds)>0:
        #     cH, cV, cD = self.coeffs[0]
        #     edge_img = pywt.idwt2([edge_img, (cH*main_mask, cV*main_mask, cD*main_mask)], wavelet=self.wavelets[i])

        edge_img = np.abs(edge_img)
        edge_img = (edge_img>=(self.binary_threshold.get()))
        edge_img = skeletonize(edge_img)

        # if len(self.thresholds)>0:
        #     edge_img = self.coherences[0]
        # edge_img = main_mask
        # edge_img = (edge_img>=(self.binary_threshold.get()))
        # edge_img = skeletonize(edge_img)
        # more processing

        edge_img = Image.fromarray((255/(np.max(edge_img)+1e-8)*edge_img).clip(0, 255).astype(np.uint8))
        edge_img.thumbnail((self.max_width, self.max_height), Image.LANCZOS)

        # update the right‐hand canvas
        self.edge_canvas.delete("all")
        self.tk_edge = ImageTk.PhotoImage(edge_img)
        self.edge_canvas.create_image(0, 0, anchor="nw", image=self.tk_edge)


app = WaveletEdgeApp()
app.mainloop()
