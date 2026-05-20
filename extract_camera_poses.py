"""
extract_camera_poses.py – Blender script: read current rigged camera poses.

Reads each rigged camera's current location and orientation from the scene
and prints them as a Python dict you can paste directly into config.py as
overrides — or feed straight into the pipeline as starting values.

Usage
─────
1.  Open in Blender (Scripting tab → Open).
2.  Run the script (▶).
3.  Copy the printed output from the System Console
    (Window → Toggle System Console on Windows).

Output format
─────────────
CAMERA_POSE_OVERRIDES = {
    '04681': {'x':  0.00, 'y':  0.00, 'z':  3.60,
              'heading': 145.40, 'pitch':  1.90, 'roll':  7.90},
    ...
}

These values are in ENU world space and match the conventions used by
pose.py and orientation_solver.py:
  x, y, z   – East, North, Up in metres (ENU)
  heading   – compass bearing in degrees (0=N, 90=E, clockwise)
  pitch     – degrees, negative = looking down
  roll      – degrees, positive = roll right
"""

import bpy
import math


def angle_wrap(deg):
    """Wrap angle to (-180, 180]."""
    return (deg + 180) % 360 - 180


print()
print("CAMERA_POSE_OVERRIDES = {")

for obj in sorted(bpy.data.objects, key=lambda o: o.name):
    # Only top-level rigged camera empties (name is just the frame number)
    if obj.parent is not None:
        continue
    if obj.type != 'EMPTY':
        continue
    # Frame numbers are 5 digits; skip other empties (Ground, etc.)
    name = obj.name
    if not name.isdigit() or len(name) != 5:
        continue

    # ── Position ──────────────────────────────────────────────────────────────
    x = obj.location.x
    y = obj.location.y
    z = obj.location.z

    # ── Heading — from yaw empty ───────────────────────────────────────────────
    # export_blender.py sets: yaw.rotation_euler[2] = radians(-heading)
    yaw_obj = bpy.data.objects.get(f'yaw.{name}')
    if yaw_obj is None:
        print(f"    # WARNING: yaw.{name} not found — skipping")
        continue
    heading = -math.degrees(yaw_obj.rotation_euler[2])
    heading = heading % 360   # normalise to [0, 360)

    # ── Pitch — from pitch empty ───────────────────────────────────────────────
    # export_blender.py sets: pitch.rotation_euler[0] = radians(pitch)
    pitch_obj = bpy.data.objects.get(f'pitch.{name}')
    if pitch_obj is None:
        print(f"    # WARNING: pitch.{name} not found — skipping")
        continue
    pitch = math.degrees(pitch_obj.rotation_euler[0])

    # ── Roll — from camera object ──────────────────────────────────────────────
    # export_blender.py sets: cam.rotation_euler[1] = radians(roll)
    cam_obj = bpy.data.objects.get(f'Camera.{name}')
    if cam_obj is None:
        print(f"    # WARNING: Camera.{name} not found — skipping")
        continue
    roll = math.degrees(cam_obj.rotation_euler[1])

    print(f"    '{name}': {{'x': {x:8.3f}, 'y': {y:8.3f}, 'z': {z:8.3f},")
    print(f"              'heading': {heading:7.2f}, 'pitch': {pitch:7.2f}, 'roll': {roll:7.2f}}},")

print("}")
print()
