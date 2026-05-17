"""
manual_calibrate.py – Manual ground-feature correspondence → pitch calibration.

Usage
─────
  python manual_calibrate.py --frames_dir ..\C1_TEST2

Shows the two undistorted frames side by side.  Click alternating points:
  1. Click a distinctive ground feature in the LEFT frame  (green dot)
  2. Click the SAME feature in the RIGHT frame             (red dot)
  3. Repeat for 4–10 feature pairs.
  4. Press ENTER or close the window to run the optimizer.
  Press U to undo the last pair.

The optimised pitch for each camera is printed and written to
pitch_overrides.txt so you can paste it into config.py.
"""

import argparse
import sys
import os
import math

import matplotlib
matplotlib.use('TkAgg')          # works on Windows; fall back to default if needed
import matplotlib.pyplot as plt
import numpy as np

# ── pipeline imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from pipeline.frame      import Frame
from pipeline.ocr        import extract_telemetry_all
from pipeline.undistort  import undistort_all
from pipeline.pose       import estimate_poses
from pitch_optimizer     import optimize_pitches

import config


def run_pipeline(frames_dir: str) -> list[Frame]:
    """Run OCR → undistort → pose (no refine)."""
    from pipeline.frame import load_frames
    frames = load_frames(frames_dir)
    extract_telemetry_all(frames)
    undistort_all(frames)
    estimate_poses(frames)
    return frames


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True)
    args = parser.parse_args()

    frames = run_pipeline(args.frames_dir)
    ready  = [f for f in frames if f.undistorted is not None]

    if len(ready) != 2:
        print(f"ERROR: put exactly 2 frames in --frames_dir (found {len(ready)}).")
        sys.exit(1)

    fa, fb = ready
    img_a  = fa.undistorted[:, :, ::-1]   # BGR → RGB
    img_b  = fb.undistorted[:, :, ::-1]

    print(f"\nFrame A: {fa.stem}  pos={fa.position_enu}  hdg={fa.heading_deg}°  roll={fa.camera_roll_deg:.1f}°")
    print(f"Frame B: {fb.stem}  pos={fb.position_enu}  hdg={fb.heading_deg}°  roll={fb.camera_roll_deg:.1f}°")
    print(f"\nClick a ground feature in LEFT image, then the SAME feature in RIGHT image.")
    print(f"Repeat 4–10 times.  Press ENTER when done.  Press U to undo.\n")

    # ── Interactive figure ────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Click matching ground features   |   ENTER = run optimiser   |   U = undo",
                 fontsize=12, fontweight='bold')

    ax_a.imshow(img_a)
    ax_a.set_title(f"A: {fa.stem[-16:]}  hdg={fa.heading_deg:.0f}°  agl={fa.position_enu[2]:.1f}m",
                   color='steelblue')
    ax_a.axis('off')

    ax_b.imshow(img_b)
    ax_b.set_title(f"B: {fb.stem[-16:]}  hdg={fb.heading_deg:.0f}°  agl={fb.position_enu[2]:.1f}m",
                   color='tomato')
    ax_b.axis('off')
    plt.tight_layout()

    pts_a:  list[tuple[float, float]] = []
    pts_b:  list[tuple[float, float]] = []
    state   = [0]          # 0 = waiting for A, 1 = waiting for B
    markers = []           # (artist_a, artist_b) tuples for undo

    def _label(ax, x, y, n, color):
        dot,  = ax.plot(x, y, 'o', color=color, markersize=9, markeredgecolor='white', lw=1.5)
        txt   = ax.text(x + 8, y - 8, str(n), color=color, fontsize=9, fontweight='bold')
        return dot, txt

    def on_click(event):
        if event.button != 1 or event.xdata is None:
            return
        n = len(pts_a) + 1

        if event.inaxes is ax_a and state[0] == 0:
            pts_a.append((float(event.xdata), float(event.ydata)))
            d, t = _label(ax_a, event.xdata, event.ydata, n, 'lime')
            markers.append({'a': (d, t), 'b': None})
            state[0] = 1
            ax_a.set_title(f"A: {fa.stem[-16:]}  ← now click the SAME point in B →",
                           color='steelblue')
            fig.canvas.draw_idle()

        elif event.inaxes is ax_b and state[0] == 1:
            pts_b.append((float(event.xdata), float(event.ydata)))
            d, t = _label(ax_b, event.xdata, event.ydata, len(pts_b), 'tomato')
            markers[-1]['b'] = (d, t)
            state[0] = 0
            ax_a.set_title(f"A: {fa.stem[-16:]}  hdg={fa.heading_deg:.0f}°  agl={fa.position_enu[2]:.1f}m",
                           color='steelblue')
            print(f"  Pair {len(pts_a)}: A=({pts_a[-1][0]:.0f},{pts_a[-1][1]:.0f})  "
                  f"B=({pts_b[-1][0]:.0f},{pts_b[-1][1]:.0f})")
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'u' and markers:
            m = markers.pop()
            for art in (m['a'] or []) + (m['b'] or []):
                if art:
                    art.remove()
            if state[0] == 0 and pts_b:
                pts_a.pop(); pts_b.pop()
            elif state[0] == 1 and pts_a:
                pts_a.pop()
                state[0] = 0
            fig.canvas.draw_idle()
            print("  Undone last point.")

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # ── Run optimiser ─────────────────────────────────────────────────────────
    if len(pts_a) < 3:
        print("Need at least 3 pairs. Exiting.")
        return

    print(f"\nRunning optimiser with {len(pts_a)} manual correspondences…")
    print(f"Initial pitch = 0.0° for both cameras.\n")

    cameras = [
        {'pos': fa.position_enu.copy(), 'yaw': fa.heading_deg,
         'roll': fa.camera_roll_deg, 'K': fa.K_undist},
        {'pos': fb.position_enu.copy(), 'yaw': fb.heading_deg,
         'roll': fb.camera_roll_deg, 'K': fb.K_undist},
    ]
    features = [
        {0: (ua, va), 1: (ub, vb)}
        for (ua, va), (ub, vb) in zip(pts_a, pts_b)
    ]

    pitches, result = optimize_pitches(
        cameras, features,
        initial_pitches=[0.0, 0.0],
        z_ground=0.0,
        verbose=True,
    )

    pf = result.cost / max(len(features), 1)
    print(f"\n{'─'*60}")
    print(f"  per-feature cost : {pf:.2f} m²  (√ = {math.sqrt(pf):.2f} m avg error)")
    print(f"  nfev             : {result.nfev}")
    print(f"  status           : {result.message}")
    print(f"{'─'*60}")
    print(f"  Frame A  {fa.stem[-16:]}  pitch = {pitches[0]:+.2f}°")
    print(f"  Frame B  {fb.stem[-16:]}  pitch = {pitches[1]:+.2f}°")
    print(f"{'─'*60}\n")

    # Write overrides for easy copy-paste into config.py
    out = os.path.join(os.path.dirname(__file__), 'pitch_overrides.txt')
    with open(out, 'w') as fh:
        fh.write("# Paste into config.py  GIMBAL_PITCH_OVERRIDES\n")
        fh.write(f'    "{fa.stem}": {pitches[0]:.2f},\n')
        fh.write(f'    "{fb.stem}": {pitches[1]:.2f},\n')
    print(f"Overrides written to {out}")


if __name__ == '__main__':
    main()
