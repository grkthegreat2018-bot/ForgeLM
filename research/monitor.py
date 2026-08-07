"""Tkinter GUI monitor for distillation/training progress.

Features:
  - Progress bar, ETA, loss breakdown, throughput, VRAM
  - Live loss chart with color-coded segments (save points highlighted)
  - Save-point indicators (vertical markers + colored segments)
  - Settings tab: font size, window size, refresh interval, chart colors
  - Fully resizable + fullscreen (F11)

Reads status.json written by distill.py. Open/close anytime without affecting training.

Usage:
    python -m research.monitor
    python -m research.monitor --status research/checkpoints/distill_status.json
"""
import argparse
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import tkinter as tk
from tkinter import ttk


class MonitorGUI:
    def __init__(self, root, status_file):
        self.root = root
        self.status_file = status_file
        self.loss_history = []  # list of (loss, is_save_point)
        self._last_step = -1
        self.max_history = 500
        self.last_status = None
        self._dirty = True
        self._smooth_lo = None
        self._smooth_hi = None
        # Persistent canvas object IDs (updated via coords/itemconfig, not recreated).
        self._poly_id = None
        self._line_id = None
        self._label_max_id = None
        self._label_min_id = None
        self._label_now_id = None
        # Scroll-compress animation state (heart-monitor sweep).
        self._animating = False
        self._anim_start = time.time()
        self._anim_duration = 0.5
        self._anim_progress = 1.0

        # Settings (persisted to ~/.forgeai_monitor.json).
        self.settings = self.load_settings()
        self.font_family = "Consolas"
        # Now that settings are loaded, sync animation duration to refresh interval.
        self._anim_duration = self.settings["refresh_ms"] / 1000.0

        root.title("ForgeAI Distillation Monitor")
        self.apply_window_size()
        root.minsize(400, 350)
        root.resizable(True, True)

        # Notebook: Monitor + Settings tabs.
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.monitor_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.monitor_tab, text="Monitor")
        self.notebook.add(self.settings_tab, text="Settings")

        self.build_monitor_tab()
        self.build_settings_tab()

        # Fullscreen bindings.
        root.bind("<F11>", self.toggle_fullscreen)
        root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self._fullscreen = False

        # Resize handler -> mark dirty.
        self.canvas.bind("<Configure>", lambda e: self._mark_dirty())

        self.refresh()
        self._start_continuous_loop()

    # ---------- Settings ----------
    def load_settings(self):
        path = os.path.expanduser("~/.forgeai_monitor.json")
        defaults = {
            "font_size": 10, "window_w": 600, "window_h": 500,
            "refresh_ms": 2000, "line_color": "#4ec9b0",
            "save_color": "#f0a060", "save_marker_color": "#ffd700",
            "bg_color": "#1e1e1e",
        }
        try:
            with open(path) as f:
                defaults.update(json.load(f))
        except Exception:
            pass
        return defaults

    def save_settings(self):
        path = os.path.expanduser("~/.forgeai_monitor.json")
        try:
            with open(path, "w") as f:
                json.dump(self.settings, f)
        except Exception:
            pass

    def apply_window_size(self):
        w = self.settings.get("window_w", 600)
        h = self.settings.get("window_h", 500)
        self.root.geometry(f"{w}x{h}")

    def font(self, size=None, bold=False):
        sz = size or self.settings["font_size"]
        return (self.font_family, sz, "bold") if bold else (self.font_family, sz)

    # ---------- Monitor tab ----------
    def build_monitor_tab(self):
        t = self.monitor_tab
        t.columnconfigure(0, weight=1)
        t.rowconfigure(3, weight=1)

        # Status frame
        sf = ttk.LabelFrame(t, text="Status", padding=10)
        sf.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        sf.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(sf, text="Loading...", font=self.font(12))
        self.status_label.grid(row=0, column=0, sticky="w")
        self.save_indicator = ttk.Label(sf, text="", font=self.font(9))
        self.save_indicator.grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Progress bar
        pf = ttk.LabelFrame(t, text="Progress", padding=10)
        pf.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        pf.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(pf, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.prog_label = ttk.Label(pf, text="0%", font=self.font(11))
        self.prog_label.grid(row=1, column=0, pady=(5, 0))

        # Stats grid
        mf = ttk.LabelFrame(t, text="Metrics", padding=10)
        mf.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        for c in range(6):
            mf.columnconfigure(c, weight=1 if c % 2 == 1 else 0)
        self.stats = {}
        metrics = [
            ("Step", "step"), ("Remaining", "remaining"), ("ETA", "eta"),
            ("Loss", "loss"), ("KL", "kl"), ("CE", "ce"),
            ("LR", "lr"), ("Throughput", "tok_s"), ("VRAM", "vram"),
        ]
        for i, (label, key) in enumerate(metrics):
            row, col = divmod(i, 3)
            ttk.Label(mf, text=label + ":", font=self.font(bold=True)).grid(
                row=row, column=col * 2, sticky="w", padx=(0, 5), pady=3)
            self.stats[key] = ttk.Label(mf, text="—", font=self.font())
            self.stats[key].grid(row=row, column=col * 2 + 1, sticky="w", pady=3)

        # Chart
        cf = ttk.LabelFrame(t, text="Loss Trend", padding=5)
        cf.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(cf, bg=self.settings["bg_color"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Footer
        self.footer = ttk.Label(t, text=f"Monitoring: {self.status_file}",
                                font=self.font(8), foreground="gray")
        self.footer.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 5))

    # ---------- Settings tab ----------
    def build_settings_tab(self):
        t = self.settings_tab
        t.columnconfigure(1, weight=1)
        row = 0

        def add_slider(label, key, lo, hi, fmt="{:.0f}"):
            nonlocal row
            ttk.Label(t, text=label, font=self.font(bold=True)).grid(
                row=row, column=0, sticky="w", padx=10, pady=8)
            var = tk.DoubleVar(value=self.settings[key])
            s = ttk.Scale(t, from_=lo, to=hi, variable=var, orient="horizontal",
                          length=200)
            s.grid(row=row, column=1, sticky="ew", padx=10)
            lbl = ttk.Label(t, text=fmt.format(self.settings[key]), font=self.font())
            lbl.grid(row=row, column=2, padx=10)
            def on_change(_=None):
                self.settings[key] = var.get()
                lbl.config(text=fmt.format(var.get()))
                self._mark_dirty()
            s.config(command=on_change)
            row += 1
            return var, lbl

        add_slider("Font size", "font_size", 8, 20)
        add_slider("Window width", "window_w", 400, 2000)
        add_slider("Window height", "window_h", 350, 2000)
        add_slider("Refresh (ms)", "refresh_ms", 500, 10000)

        # Color pickers (simple entry fields).
        ttk.Separator(t, orient="horizontal").grid(row=row, column=0, columnspan=3,
                                                    sticky="ew", padx=10, pady=10)
        row += 1
        ttk.Label(t, text="Colors (hex)", font=self.font(bold=True)).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=10)
        row += 1
        for label, key in [("Line", "line_color"), ("Save segment", "save_color"),
                           ("Save marker", "save_marker_color"), ("Background", "bg_color")]:
            ttk.Label(t, text=label, font=self.font()).grid(
                row=row, column=0, sticky="w", padx=10, pady=4)
            entry = ttk.Entry(t, width=12, font=self.font())
            entry.insert(0, self.settings[key])
            entry.grid(row=row, column=1, sticky="w", padx=10)
            def make_setter(k, e):
                def setter(_=None):
                    self.settings[k] = e.get()
                    if k == "bg_color":
                        self.canvas.config(bg=e.get())
                    self._mark_dirty()
                return setter
            entry.bind("<FocusOut>", make_setter(key, entry))
            entry.bind("<Return>", make_setter(key, entry))
            row += 1

        # Apply + Save buttons.
        btn_frame = ttk.Frame(t)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        ttk.Button(btn_frame, text="Apply & Save", command=self.apply_settings).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_settings).pack(side="left", padx=5)

    def apply_settings(self):
        self.root.geometry(f"{int(self.settings['window_w'])}x{int(self.settings['window_h'])}")
        self.canvas.config(bg=self.settings["bg_color"])
        self.save_settings()
        self._mark_dirty()

    def reset_settings(self):
        path = os.path.expanduser("~/.forgeai_monitor.json")
        try:
            os.remove(path)
        except Exception:
            pass
        self.root.destroy()

    # ---------- Refresh (data polling) ----------
    def refresh(self):
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file) as f:
                    s = json.load(f)
                self.last_status = s
                self.update_display(s)
            else:
                self.status_label.config(text="Waiting for training to start... (no status file yet)",
                                         foreground="orange")
        except (json.JSONDecodeError, IOError):
            self.status_label.config(text="Reading status... (file being written)",
                                     foreground="orange")
        # Rendering is handled by the continuous 60fps loop (_start_continuous_loop).
        self.root.after(int(self.settings["refresh_ms"]), self.refresh)

    def _mark_dirty(self):
        self._dirty = True

    def update_display(self, s):
        step = s.get("step", 0)
        total = s.get("total_steps", 0)
        pct = s.get("pct", 0)
        running = s.get("running", False)
        finished = s.get("finished", False)
        is_save = step % 100 == 0 and step > 0 and running  # save_every=100

        if finished:
            self.status_label.config(text="✓ Training COMPLETE!", foreground="green")
        elif running:
            self.status_label.config(text="● Training in progress", foreground="green")
        else:
            self.status_label.config(text="○ Paused or stopped", foreground="orange")

        # Save indicator.
        if is_save:
            self.save_indicator.config(text=f"💾 Checkpoint saved at step {step}",
                                        foreground=self.settings["save_marker_color"])
        else:
            self.save_indicator.config(text="")

        self.progress["value"] = pct
        self.prog_label.config(text=f"{step:,} / {total:,}  ({pct:.1f}%)")

        eta_h = s.get("eta_hours", 0)
        eta_m = s.get("eta_minutes", 0)
        self.stats["step"].config(text=f"{step:,} / {total:,}")
        self.stats["remaining"].config(text=f"{s.get('remaining', 0):,}")
        self.stats["eta"].config(text=f"{eta_h}h {eta_m:02d}m")
        self.stats["loss"].config(text=f"{s.get('loss', 0):.4f}")
        self.stats["kl"].config(text=f"{s.get('kl', 0):.4f}")
        self.stats["ce"].config(text=f"{s.get('ce', 0):.4f}")
        self.stats["lr"].config(text=f"{s.get('lr', 0):.2e}")
        self.stats["tok_s"].config(text=f"{s.get('tok_s', 0):.0f}")
        self.stats["vram"].config(text=f"{s.get('vram_gb', 0):.2f} GB")

        loss = s.get("loss", 0)
        if loss > 0:
            # Only add a new point when the STEP changes (not just re-reading same status).
            step_changed = not self.loss_history or self._last_step != step
            if step_changed:
                self._last_step = step
                self.loss_history.append((loss, is_save))
                if len(self.loss_history) > self.max_history:
                    self.loss_history.pop(0)
                # Smooth continuous transition spanning the full polling interval
                self._anim_start = time.time()
                self._anim_duration = max(0.1, self.settings["refresh_ms"] / 1000.0)
                self._animating = True
                self._mark_dirty()

    # ---------- Render (60 FPS continuous loop) ----------
    def _start_continuous_loop(self):
        """Continuous ~30fps loop for fluid chart compression and scale smoothing.
        Wrapped in try/except so a single frame error doesn't kill the loop."""
        try:
            self._update_frame()
        except Exception as e:
            import traceback
            traceback.print_exc()
        # Always reschedule, even if _update_frame threw — prevents the loop from dying.
        self.root.after(33, self._start_continuous_loop)  # ~30fps

    def _update_frame(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10 or not self.loss_history:
            return

        losses = [l for l, _ in self.loss_history]
        raw_lo, raw_hi = min(losses), max(losses)
        if raw_hi == raw_lo:
            raw_hi = raw_lo + 1.0
        pad_range = (raw_hi - raw_lo) * 0.10
        target_lo = raw_lo - pad_range
        target_hi = raw_hi + pad_range

        # Frame-by-frame continuous exponential smoothing for Y scale limits
        if self._smooth_lo is None:
            self._smooth_lo = target_lo
            self._smooth_hi = target_hi
        else:
            # Slower smoothing factor (0.02) for a gentle, cinematic "smooth zoom out"
            self._smooth_lo += (target_lo - self._smooth_lo) * 0.02
            self._smooth_hi += (target_hi - self._smooth_hi) * 0.02

        # Calculate progress spanning full update window
        if self._animating:
            elapsed = time.time() - self._anim_start
            t = min(1.0, elapsed / self._anim_duration)
            self._anim_progress = t
            if t >= 1.0:
                self._animating = False

        # Always redraw — the skip-when-idle optimization caused apparent freezing.
        self._draw_chart(w, h, self._smooth_lo, self._smooth_hi)
        self._dirty = False

    def render(self):
        self._update_frame()

    def _y_pos(self, val, lo, hi, plot_h):
        """Map a loss value to a y-coordinate on the canvas."""
        return plot_h - (val - lo) / (hi - lo) * (plot_h - 15) - 5

    def _draw_chart(self, w, h, lo, hi):
        """Draw the chart with continuous leftward flow (heart-monitor style).

        Animation model:
        - All N points always spread evenly across full width, becoming more compressed
          as N increases.
        - When a new point arrives, we animate the shift of existing points to their
          new, slightly more compressed X positions over the polling interval.
        """
        if len(self.loss_history) < 1:
            if self._label_now_id is None:
                self._label_now_id = self.canvas.create_text(
                    w // 2, h // 2, text="Waiting for data...",
                    fill="#888", font=self.font(14), tags="waiting")
            return
        self.canvas.delete("waiting")

        if len(self.loss_history) < 2:
            # Draw a single line across the screen for the first point.
            losses = [l for l, _ in self.loss_history]
            y = self._y_pos(losses[0], lo, hi, h - 25)
            if getattr(self, "_line_id", None) is None:
                self._line_id = self.canvas.create_line(
                    -50, y, w + 50, y, fill=self.settings["line_color"], width=2, tags="line")
            else:
                self.canvas.coords(self._line_id, -50, y, w + 50, y)
            return

        losses = [l for l, _ in self.loss_history]
        save_flags = [s for _, s in self.loss_history]
        n = len(losses)

        # Use full width to avoid any horizontal gaps on the right side
        plot_w = w
        plot_h = h - 25  # Keep vertical padding for bottom labels
        line_color = self.settings["line_color"]
        save_color = self.settings.get("save_color", "#f0a060")
        fill_color = "#233a36"
        fs = max(8, min(14, h // 25))

        p = self._anim_progress if self._animating else 1.0

        # Before (N-1 points) and after (N points) step sizes.
        step_before = plot_w / (n - 2) if n > 2 else plot_w
        step_after = plot_w / (n - 1)

        new_y_final = self._y_pos(losses[-1], lo, hi, plot_h)
        prev_y = self._y_pos(losses[-2], lo, hi, plot_h)
        new_y = prev_y + (new_y_final - prev_y) * p

        # Compute all point positions (x, y) into a flat list.
        coords = []
        points = []  # list of (x, y) pairs for save-segment logic
        for i in range(n - 1):
            x_before = i * step_before
            x_after = i * step_after
            x = x_before + (x_after - x_before) * p
            y = self._y_pos(losses[i], lo, hi, plot_h)
            coords.extend([x, y])
            points.append((x, y))
        # Newest point at far right edge.
        coords.extend([plot_w, new_y])
        points.append((plot_w, new_y))

        # Polygon for area fill: bottom-left -> line points -> bottom-right.
        # Use plot_h (not h+50) for bottom to avoid off-screen artifacts.
        poly_coords = [points[0][0], plot_h]
        for x, y in points:
            poly_coords.extend([x, y])
        poly_coords.extend([points[-1][0], plot_h])

        # Create or update the polygon and line objects.
        if getattr(self, "_poly_id", None) is None:
            self._poly_id = self.canvas.create_polygon(
                *poly_coords, fill=fill_color, outline="", tags="poly")
            self._line_id = self.canvas.create_line(
                *coords, fill=line_color, width=2, tags="line",
                capstyle="round", joinstyle="round")
        else:
            self.canvas.coords(self._poly_id, *poly_coords)
            self.canvas.itemconfig(self._poly_id, fill=fill_color)
            self.canvas.coords(self._line_id, *coords)
            self.canvas.itemconfig(self._line_id, fill=line_color, width=2)

        # Draw save segments as flat orange overlays at the save point's height.
        # Flat = both endpoints at save Y, so it reads as part of the main line,
        # not a separate overlapping trace.
        self.canvas.delete("save_seg")
        for i in range(1, n):
            if save_flags[i]:
                x0, _ = points[i - 1]
                x1, y1 = points[i]
                save_y = y1  # flat at the save point's height
                self.canvas.create_line(
                    x0, save_y, x1, save_y, fill=save_color, width=3,
                    capstyle="round", joinstyle="round", tags="save_seg")

        # Update labels via itemconfig (no delete/recreate flicker).
        if self._label_max_id is None:
            self._label_max_id = self.canvas.create_text(
                10, 5, text=f"max {hi:.2f}", fill="#888", anchor="nw",
                font=self.font(fs), tags="label_max")
            self._label_min_id = self.canvas.create_text(
                10, h - 5, text=f"min {lo:.2f}", fill="#888", anchor="sw",
                font=self.font(fs), tags="label_min")
            self._label_now_id = self.canvas.create_text(
                w - 5, 5, text=f"now {losses[-1]:.2f}", fill=line_color,
                anchor="ne", font=self.font(fs), tags="label_now")
        else:
            self.canvas.itemconfig(self._label_max_id, text=f"max {hi:.2f}",
                                   font=self.font(fs))
            self.canvas.coords(self._label_min_id, 10, h - 5)
            self.canvas.itemconfig(self._label_min_id, text=f"min {lo:.2f}",
                                   font=self.font(fs))
            self.canvas.coords(self._label_now_id, w - 5, 5)
            self.canvas.itemconfig(self._label_now_id,
                                   text=f"now {losses[-1]:.2f}",
                                   fill=line_color, font=self.font(fs))

    # ---------- Fullscreen ----------
    def toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)
        # Reset canvas object IDs since grid_forget clears the canvas.
        self.canvas.delete("all")
        self._poly_id = None
        self._line_id = None
        self._label_max_id = None
        self._label_min_id = None
        self._label_now_id = None
        # Force a complete redraw with new geometry.
        self.root.update_idletasks()
        self._mark_dirty()
        self.render()


def main():
    p = argparse.ArgumentParser(description="GUI monitor for distillation progress")
    p.add_argument("--status", default="research/checkpoints/distill_status.json",
                   help="Path to status.json file written by distill.py")
    args = p.parse_args()
    root = tk.Tk()
    MonitorGUI(root, args.status)
    root.mainloop()


if __name__ == "__main__":
    main()
