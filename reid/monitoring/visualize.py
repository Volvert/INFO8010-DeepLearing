# This is a visualization tool of vehicle re-identification.
# Step 1: Save the matched test image IDs for each query image as a text file, where each line contains a test image ID ranked in terms of distance score in ascending order. Name each text file as '%06d.txt' % <query_image_ID>. We assume that the top-50 matched test images are displayed. An example is given in "./dist_example/".
# Step 2: Run "python visualize.py".
# Step 3: Input the path of the directory containing all text files at "Txt Dir:" (end with '/'). An example is given as "./dist_example/".
# Step 4: Click "Load".
# The query image is shown on the top left. The corresponding test images are shown on the right.
# For each image, the image ID is shown on the top left corner.
# Click "<< Prev" to return to the previous query.
# Click "Next >>" to advance to the next query.
# Enter the query no. and click "Go" to jump to the corresponding query.

import os
import glob
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk, ImageDraw


THUMB_SIZE    = (150, 150)
GALLERY_COLS  = 10
GALLERY_ROWS  = 5
GALLERY_W     = THUMB_SIZE[0] * GALLERY_COLS
GALLERY_H     = THUMB_SIZE[1] * GALLERY_ROWS
LABEL_COLOR   = (255, 165, 0)


def _open_image(path: str | Path, size: tuple[int, int]) -> Image.Image:
    try:
        return Image.open(str(path)).convert("RGB").resize(size, Image.LANCZOS)
    except (FileNotFoundError, OSError):
        placeholder = Image.new("RGB", size, color=(180, 180, 180))
        draw = ImageDraw.Draw(placeholder)
        draw.line([(0, 0), size], fill=(120, 120, 120), width=2)
        draw.line([(size[0], 0), (0, size[1])], fill=(120, 120, 120), width=2)
        return placeholder


def _overlay_id(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (len(label) * 8 + 4, 18)], fill=(0, 0, 0, 160))
    draw.text((3, 2), label, fill=LABEL_COLOR)
    return img


class VisTool:

    def __init__(self, master: tk.Tk) -> None:
        self.parent = master
        self.parent.title("VisTool")

        self.txt_dir:    str       = ""
        self.txt_list:   list[str] = []
        self.cur:        int       = 0
        self.total:      int       = 0
        self._tkimg:     Optional[ImageTk.PhotoImage] = None
        self._prb_tkimg: Optional[ImageTk.PhotoImage] = None

        self._build_widgets()
        self._center_window(width=GALLERY_W + 200, height=GALLERY_H + 120)

    def _build_widgets(self) -> None:
        self.frame = tk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # row 0 — directory entry + load
        tk.Label(self.frame, text="Txt Dir:").grid(row=0, column=0, sticky=tk.E, padx=4, pady=4)
        self.entry = tk.Entry(self.frame, width=50)
        self.entry.grid(row=0, column=1, sticky=tk.W + tk.E, padx=4)
        tk.Button(self.frame, text="Load", width=8, command=self.loadDir).grid(row=0, column=2, padx=4)

        # left — query panel
        self.prbPanel = tk.Frame(self.frame, bd=8)
        self.prbPanel.grid(row=1, column=0, rowspan=4, sticky=tk.N + tk.S)
        tk.Label(self.prbPanel, text="Query image:").pack(side=tk.TOP, pady=4)
        self.prbLabel = tk.Label(self.prbPanel, width=THUMB_SIZE[0], height=THUMB_SIZE[1])
        self.prbLabel.pack(side=tk.TOP)
        self.queryIdLabel = tk.Label(self.prbPanel, text="", fg="darkorange")
        self.queryIdLabel.pack(side=tk.TOP, pady=8)

        # center — gallery canvas
        self.mainPanel = tk.Canvas(self.frame, width=GALLERY_W, height=GALLERY_H,
                                   cursor="arrow", bg="#eeeeee")
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=tk.W + tk.N)
        self.parent.bind("<KeyPress-a>", self.prevPrb)
        self.parent.bind("<KeyPress-d>", self.nextPrb)

        # row 5 — navigation controls
        ctrl = tk.Frame(self.frame)
        ctrl.grid(row=5, column=0, columnspan=3, sticky=tk.W + tk.E, pady=4)
        tk.Button(ctrl, text="<< Prev", width=10, command=self.prevPrb).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl, text="Next >>", width=10, command=self.nextPrb).pack(side=tk.LEFT, padx=6)
        self.progLabel = tk.Label(ctrl, text="    0 /    0", width=12)
        self.progLabel.pack(side=tk.LEFT, padx=8)
        tk.Label(ctrl, text="Go to Query #").pack(side=tk.LEFT, padx=4)
        self.idxEntry = tk.Entry(ctrl, width=6)
        self.idxEntry.pack(side=tk.LEFT)
        tk.Button(ctrl, text="Go", command=self.gotoPrb).pack(side=tk.LEFT, padx=4)

        # row 6 — status bar
        self.statusVar = tk.StringVar(value="Load a directory to start.")
        tk.Label(self.frame, textvariable=self.statusVar, anchor=tk.W,
                 relief=tk.SUNKEN, bd=1).grid(row=6, column=0, columnspan=3, sticky=tk.W + tk.E)

        self.frame.columnconfigure(1, weight=1)

    def _center_window(self, width: int, height: int) -> None:
        self.parent.update_idletasks()
        x = (self.parent.winfo_screenwidth()  - width)  // 2
        y = (self.parent.winfo_screenheight() - height) // 2
        self.parent.geometry(f"{width}x{height}+{x}+{y}")

    def loadDir(self) -> None:
        self.txt_dir  = self.entry.get().strip()
        self.txt_list = sorted(
            glob.glob(os.path.join(self.txt_dir, "*.txt")),
            key=lambda p: int(Path(p).stem)
        )
        if not self.txt_list:
            messagebox.showwarning("No files found",
                                   f"No .txt files found in:\n{self.txt_dir}")
            return
        self.cur   = 1
        self.total = len(self.txt_list)
        self.loadImage()
        self.statusVar.set(f"Loaded {self.total} queries from {self.txt_dir}")

    def loadImage(self) -> None:
        txt_path  = self.txt_list[self.cur - 1]
        query_id  = Path(txt_path).stem
        base_dir  = Path(self.txt_dir)
        query_root   = base_dir.parent / "image_query"
        gallery_root = base_dir.parent / "image_test"

        # query image
        q_img = _overlay_id(_open_image(query_root / f"{query_id}.jpg", THUMB_SIZE), query_id)
        self._prb_tkimg = ImageTk.PhotoImage(q_img)
        self.prbLabel.config(image=self._prb_tkimg)
        self.queryIdLabel.config(text=f"ID: {query_id}")

        # gallery images
        ranked_ids: list[str] = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ranked_ids.append(f"{int(line):06d}")
        ranked_ids = ranked_ids[:GALLERY_COLS * GALLERY_ROWS]

        gallery_img = Image.new("RGB", (GALLERY_W, GALLERY_H), color=(238, 238, 238))
        for i, gid in enumerate(ranked_ids):
            thumb = _overlay_id(_open_image(gallery_root / f"{gid}.jpg", THUMB_SIZE), gid)
            gallery_img.paste(thumb, ((i % GALLERY_COLS) * THUMB_SIZE[0],
                                      (i // GALLERY_COLS) * THUMB_SIZE[1]))

        self._tkimg = ImageTk.PhotoImage(gallery_img)
        self.mainPanel.delete("all")
        self.mainPanel.create_image(0, 0, image=self._tkimg, anchor=tk.NW)
        self.progLabel.config(text=f"{self.cur:5d}/{self.total:5d}")
        self.statusVar.set(f"Query {self.cur}/{self.total} — ID {query_id} — "
                           f"{len(ranked_ids)} results")

    def prevPrb(self, event: Optional[tk.Event] = None) -> None:
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextPrb(self, event: Optional[tk.Event] = None) -> None:
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoPrb(self) -> None:
        raw = self.idxEntry.get().strip()
        if not raw.isdigit():
            messagebox.showerror("Invalid input", f"'{raw}' is not a valid number.")
            return
        idx = int(raw)
        if 1 <= idx <= self.total:
            self.cur = idx
            self.loadImage()
        else:
            messagebox.showerror("Out of range",
                                 f"Query number must be between 1 and {self.total}.")


if __name__ == "__main__":
    root = tk.Tk()
    VisTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()