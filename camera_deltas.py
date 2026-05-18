"""
camera_deltas.py – Blender script: compare current rigged camera poses to export.

Usage
─────
1.  Open this script in Blender (Scripting tab → Open).
2.  Paste the FRAMES list from your blender_scene.py into ORIGINAL_FRAMES below.
3.  Run the script (▶).
4.  Results are printed to the Blender System Console
    (Window → Toggle System Console on Windows).

Output
──────
For each rigged camera, reports the delta between what was exported and
what you've manually adjusted in Blender:

  Δx, Δy, Δz      – position change in metres  (ENU: East, North, Up)
  Δheading         – yaw change in degrees
  Δpitch           – pitch change in degrees
  Δroll            – roll change in degrees

A final summary block prints the corrections as a Python dict suitable
for pasting into code.
"""

import bpy
import math

# ─────────────────────────────────────────────────────────────────────────────
# PASTE THE FRAMES LIST FROM YOUR blender_scene.py HERE
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_FRAMES = [
    # Example — replace with your actual exported data:
    # {
    #     'frame_num': '04681',
    #     'x': 0.0, 'y': 0.0, 'z': 3.6,
    #     'heading': 130.0, 'pitch': 2.0, 'roll': 7.2,
    #     'fov_deg': 59.5, 'img_path': '...', 'R': [...],
    # },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        print(f"  WARNING: object '{name}' not found in scene.")
    return obj


def angle_diff(a, b):
    """Smallest signed difference between two angles in degrees."""
    d = (a - b + 180) % 360 - 180
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if not ORIGINAL_FRAMES:
    print("ERROR: ORIGINAL_FRAMES is empty — paste your FRAMES list from blender_scene.py.")
else:
    print()
    print("=" * 72)
    print(f"  {'Frame':<8}  {'Δx(E)':>7}  {'Δy(N)':>7}  {'Δz(U)':>7}  "
          f"{'Δhdg':>7}  {'Δpitch':>7}  {'Δroll':>7}")
    print("=" * 72)

    corrections = {}

    for orig in ORIGINAL_FRAMES:
        n = orig['frame_num']

        # ── Locate Blender objects ────────────────────────────────────────────
        top   = get_obj(n)
        yaw   = get_obj(f'yaw.{n}')
        pitch = get_obj(f'pitch.{n}')
        cam   = get_obj(f'Camera.{n}')

        if None in (top, yaw, pitch, cam):
            print(f"  {n:<8}  SKIP – one or more objects missing")
            continue

        # ── Current values ────────────────────────────────────────────────────
        cur_x   = top.location.x
        cur_y   = top.location.y
        cur_z   = top.location.z

        # yaw empty: rotation_euler[2] = -heading_rad  (set as -heading in export)
        cur_hdg   = -math.degrees(yaw.rotation_euler[2])
        # pitch empty: rotation_euler[0] = pitch_rad
        cur_pitch = math.degrees(pitch.rotation_euler[0])
        # camera: rotation_euler[1] = roll_rad
        cur_roll  = math.degrees(cam.rotation_euler[1])

        # ── Deltas ────────────────────────────────────────────────────────────
        dx     = cur_x   - orig['x']
        dy     = cur_y   - orig['y']
        dz     = cur_z   - orig['z']
        d_hdg  = angle_diff(cur_hdg,   orig['heading'])
        d_pit  = angle_diff(cur_pitch, orig['pitch'])
        d_roll = angle_diff(cur_roll,  orig['roll'])

        print(f"  {n:<8}  {dx:+7.2f}  {dy:+7.2f}  {dz:+7.2f}  "
              f"{d_hdg:+7.2f}  {d_pit:+7.2f}  {d_roll:+7.2f}")

        corrections[n] = {
            'dx': round(dx, 3), 'dy': round(dy, 3), 'dz': round(dz, 3),
            'd_heading': round(d_hdg, 2),
            'd_pitch':   round(d_pit, 2),
            'd_roll':    round(d_roll, 2),
        }

    print("=" * 72)
    print()

    # ── Magnitude summary ─────────────────────────────────────────────────────
    if corrections:
        pos_mags = [
            math.sqrt(c['dx']**2 + c['dy']**2 + c['dz']**2)
            for c in corrections.values()
        ]
        print(f"Position move magnitude:  "
              f"min={min(pos_mags):.2f}m  "
              f"max={max(pos_mags):.2f}m  "
              f"avg={sum(pos_mags)/len(pos_mags):.2f}m")
        print()

    # ── Corrections dict (paste into code) ────────────────────────────────────
    print("CORRECTIONS = {")
    for n, c in corrections.items():
        print(f"    '{n}': {c},")
    print("}")
    print()
