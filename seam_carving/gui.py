"""A small Tkinter GUI for interactive seam carving.

Run with::

    python -m seam_carving.gui

The heavy work runs in a background thread so the window stays responsive.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .carve import (
    FINDERS,
    CarveResult,
    carve_height,
    carve_width,
    enlarge_height,
    enlarge_width,
)
from .visualize import overlay_horizontal_seam, overlay_vertical_seam


class SeamCarvingApp:
    """Interactive image resizing window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Seam Carving - Interactive Demo")
        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.last_overlay: np.ndarray | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.busy = False
        self._build_ui()

    def _build_ui(self) -> None:
        self.canvas = tk.Canvas(self.root, width=760, height=560, bg="#202020")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=8, pady=8)

        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        ttk.Button(toolbar, text="Open image", command=self.open_image).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(toolbar, text="Save as", command=self.save_image).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(toolbar, text="Reset", command=self.reset).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=2
        )

        ttk.Separator(toolbar).grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(toolbar, text="Method").grid(row=4, column=0, sticky="w")
        self.method = tk.StringVar(value="dp")
        ttk.Combobox(
            toolbar,
            textvariable=self.method,
            values=sorted(FINDERS),
            state="readonly",
            width=10,
        ).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(toolbar, text="Width").grid(row=5, column=0, sticky="w")
        self.width_var = tk.StringVar(value="0")
        self.width_box = ttk.Spinbox(toolbar, from_=1, to=1, textvariable=self.width_var, width=10)
        self.width_box.grid(row=5, column=1, sticky="ew", pady=2)
        self.apply_width_button = ttk.Button(
            toolbar, text="Apply width", command=self.apply_width
        )
        self.apply_width_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Label(toolbar, text="Height").grid(row=7, column=0, sticky="w")
        self.height_var = tk.StringVar(value="0")
        self.height_box = ttk.Spinbox(toolbar, from_=1, to=1, textvariable=self.height_var, width=10)
        self.height_box.grid(row=7, column=1, sticky="ew", pady=2)
        self.apply_height_button = ttk.Button(
            toolbar, text="Apply height", command=self.apply_height
        )
        self.apply_height_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=2)

        ttk.Separator(toolbar).grid(row=9, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(
            toolbar, text="Insert 1 vertical seam", command=self.insert_vertical
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(
            toolbar, text="Insert 1 horizontal seam", command=self.insert_horizontal
        ).grid(row=11, column=0, columnspan=2, sticky="ew", pady=2)

        self.show_seam = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar,
            text="Show first removed seam",
            variable=self.show_seam,
            command=self.refresh_canvas,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=8)

        self.status = tk.StringVar(value="Open an image to begin.")
        ttk.Label(toolbar, textvariable=self.status, wraplength=180).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=8
        )

        ttk.Label(
            toolbar,
            text="Tip: graphcut is exact but slower.\nUse dp for large images.",
            foreground="#666666",
            wraplength=180,
        ).grid(row=14, column=0, columnspan=2, sticky="w")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self.apply_width_button,
            self.apply_height_button,
            self.width_box,
            self.height_box,
        ):
            widget.configure(state=state)

    def open_image(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if not path:
            return
        with Image.open(path) as image:
            self.original = np.array(image.convert("RGB"))
        self.current = self.original.copy()
        self.last_overlay = None
        self._update_size_controls()
        self.refresh_canvas()
        self.status.set(f"Loaded {Path(path).name}")

    def _update_size_controls(self) -> None:
        if self.current is None:
            return
        height, width = self.current.shape[:2]
        self.width_box.configure(from_=1, to=width + 40)
        self.height_box.configure(from_=1, to=height + 40)
        self.width_var.set(str(width))
        self.height_var.set(str(height))

    def save_image(self) -> None:
        if self.current is None or self.busy:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            Image.fromarray(self.current).save(path)
            self.status.set(f"Saved {Path(path).name}")

    def reset(self) -> None:
        if self.original is None or self.busy:
            return
        self.current = self.original.copy()
        self.last_overlay = None
        self._update_size_controls()
        self.refresh_canvas()
        self.status.set("Reset to the original image.")

    def apply_width(self) -> None:
        if self.current is None or self.busy:
            return
        try:
            target = int(self.width_var.get())
        except ValueError:
            messagebox.showinfo("Invalid width", "Enter a whole number.")
            return
        if target == self.current.shape[1]:
            return
        if target > self.current.shape[1] + 40:
            messagebox.showinfo("Too large", "Insert at most 40 seams at a time.")
            return
        finder = FINDERS[self.method.get()]
        before = self.current.copy()
        if target > before.shape[1]:
            task = partial(enlarge_width, before, target, finder)
        else:
            task = partial(carve_width, before, target, finder)
        self._start_worker(task, before, overlay_vertical_seam)

    def apply_height(self) -> None:
        if self.current is None or self.busy:
            return
        try:
            target = int(self.height_var.get())
        except ValueError:
            messagebox.showinfo("Invalid height", "Enter a whole number.")
            return
        if target == self.current.shape[0]:
            return
        if target > self.current.shape[0] + 40:
            messagebox.showinfo("Too large", "Insert at most 40 seams at a time.")
            return
        finder = FINDERS[self.method.get()]
        before = self.current.copy()
        if target > before.shape[0]:
            task = partial(enlarge_height, before, target, finder)
        else:
            task = partial(carve_height, before, target, finder)
        self._start_worker(task, before, overlay_horizontal_seam)

    def insert_vertical(self) -> None:
        if self.current is None or self.busy:
            return
        finder = FINDERS[self.method.get()]
        before = self.current.copy()
        self._start_worker(
            partial(enlarge_width, before, before.shape[1] + 1, finder),
            before,
            overlay_vertical_seam,
        )

    def insert_horizontal(self) -> None:
        if self.current is None or self.busy:
            return
        finder = FINDERS[self.method.get()]
        before = self.current.copy()
        self._start_worker(
            partial(enlarge_height, before, before.shape[0] + 1, finder),
            before,
            overlay_horizontal_seam,
        )

    def _start_worker(self, task, before, overlay) -> None:
        self._set_busy(True)
        self.status.set("Working...")

        def worker() -> None:
            start = time.perf_counter()
            try:
                result: CarveResult = task()
                elapsed = time.perf_counter() - start
                last = overlay(before, result.seams[0]) if result.seams else None
                self.root.after(0, self._finish, result, elapsed, last)
            except Exception as error:  # noqa: BLE001 - shown to the user
                self.root.after(0, self._fail, error)

        threading.Thread(target=worker, daemon=True).start()

    def _finish(self, result: CarveResult, elapsed: float, last_overlay) -> None:
        self.current = result.image
        self.last_overlay = last_overlay
        self._set_busy(False)
        self._update_size_controls()
        self.refresh_canvas()
        height, width = self.current.shape[:2]
        self.status.set(f"{width}x{height} in {elapsed:.2f}s")

    def _fail(self, error: Exception) -> None:
        self._set_busy(False)
        self.status.set("Failed.")
        messagebox.showerror("Seam carving failed", str(error))

    def refresh_canvas(self) -> None:
        if self.current is None:
            return
        image = self.current
        if self.show_seam.get() and self.last_overlay is not None:
            image = self.last_overlay
        pil = Image.fromarray(image)
        scale = min(760 / pil.width, 560 / pil.height, 1.0)
        if scale < 1.0:
            pil = pil.resize(
                (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                Image.Resampling.LANCZOS,
            )
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)


def main() -> None:
    root = tk.Tk()
    SeamCarvingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
