import argparse, sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def collect_data(frames_dir, out_dir, optimize_pitch=False, enhance=False, height_mode='agl'):
    import config
    import cv2
    from pipeline.frame     import load_frames
    from pipeline.ocr       import extract_telemetry_all
    from pipeline.undistort import undistort_all
    from pipeline.pose      import detect_camera_roll

    os.makedirs(out_dir, exist_ok=True)

    from pipeline.pose import estimate_poses
    frames = load_frames(frames_dir)
    extract_telemetry_all(frames)
    undistort_all(frames)
    estimate_poses(frames)   # sets frame.R using pitch overrides / 0deg
    if optimize_pitch:
        from pipeline.refine import refine_pitches, set_enhance
        set_enhance(enhance)
        # Apply chosen height to position_enu before optimizer runs
        for f in frames:
            if f.position_enu is None: continue
            alt_ref = f.alt_takeoff_ref_m or 0.0
            alt_agl = f.alt_agl_m         or 0.0
            if height_mode == 'avg':
                f.position_enu[2] = (alt_ref + alt_agl) / 2.0
            elif height_mode == 'tor':
                f.position_enu[2] = alt_ref
            else:
                f.position_enu[2] = alt_agl
        print('Running pitch optimizer...')
        refine_pitches(frames)

    origin = next(((f.lat, f.lon) for f in frames if f.lat is not None), None)
    rows = []
    for f in frames:
        if f.lat is None or origin is None:
            continue
        frame_num = f.stem.split('_')[-1]

        # Save undistorted image
        img_filename = frame_num + '_undist.png'
        img_path = os.path.join(out_dir, img_filename)
        if f.undistorted is not None:
            cv2.imwrite(img_path, f.undistorted)

        enu   = f.position_enu if f.position_enu is not None else [0,0,0]
        alt_ref = f.alt_takeoff_ref_m or 0.0
        alt_agl = f.alt_agl_m         or 0.0
        if height_mode == 'avg':
            z = (alt_ref + alt_agl) / 2.0
        elif height_mode == 'tor':
            z = alt_ref
        else:  # 'agl' default
            z = alt_agl
        roll  = 0.0
        if f.raw is not None:
            det = detect_camera_roll(f.raw)
            if det is not None:
                roll = -det   # negate to match pose.py convention
        pitch   = f.gimbal_pitch_deg if f.gimbal_pitch_deg is not None else 0.0
        fov_deg = 90.0
        if f.K_undist is not None and f.undistorted is not None:
            fov_deg = math.degrees(2.0 * math.atan(
                f.undistorted.shape[1] / (2.0 * float(f.K_undist[0, 0]))))

        # Build full rotation matrix for app_cameras (what the app uses)
        # R is R_cam_from_world, we need camera position and orientation in world
        R = f.R.tolist() if f.R is not None else None

        rows.append(dict(
            frame_num  = frame_num,
            img_path   = os.path.abspath(img_path).replace('\\', '/'),
            x          = round(float(enu[0]), 3),
            y          = round(float(enu[1]), 3),
            z          = round(z, 3),
            heading    = round(float(f.heading_deg or 0), 2),
            pitch      = round(pitch, 2),
            roll       = round(roll, 2),
            fov_deg    = round(fov_deg, 3),
            R          = R,
        ))
    return rows


def generate_blender_script(data, out_path):
    lines = []
    A = lines.append

    A("import bpy, math, mathutils")
    A("")
    A("# --------------- helpers ---------------")
    A("")
    A("def new_empty(name, parent=None, loc=(0,0,0)):")
    A("    o = bpy.data.objects.new(name, None)")
    A("    o.empty_display_type = 'ARROWS'")
    A("    o.empty_display_size = 1.0")
    A("    bpy.context.scene.collection.objects.link(o)")
    A("    o.location = loc")
    A("    if parent: o.parent = parent")
    A("    return o")
    A("")
    A("def new_camera(name, fov_deg, img_path, parent=None):")
    A("    cd = bpy.data.cameras.new(name)")
    A("    cd.lens_unit = 'FOV'")
    A("    cd.angle = math.radians(fov_deg)")
    A("    cd.display_size = 2.0")
    A("    cd.show_name = True")
    A("    # Background image")
    A("    cd.show_background_images = True")
    A("    if img_path and os.path.exists(img_path):")
    A("        try:")
    A("            img = bpy.data.images.load(img_path)")
    A("            bg = cd.background_images.new()")
    A("            bg.image = img")
    A("            bg.alpha = 0.5")
    A("            bg.frame_method = 'FIT'")
    A("            bg.display_depth = 'FRONT'")
    A("        except Exception as e:")
    A("            print(f'Could not load {img_path}: {e}')")
    A("    co = bpy.data.objects.new(name, cd)")
    A("    bpy.context.scene.collection.objects.link(co)")
    A("    co.rotation_mode = 'XYZ'")
    A("    if parent: co.parent = parent")
    A("    return co")
    A("")
    A("def add_to_collection(obj, col):")
    A("    if obj.name not in col.objects:")
    A("        col.objects.link(obj)")
    A("    # also link all children recursively")
    A("    for child in obj.children_recursive:")
    A("        if child.name not in col.objects:")
    A("            col.objects.link(child)")
    A("")
    A("import os")
    A("")
    A("# --------------- clear scene ---------------")
    A("bpy.ops.object.select_all(action='SELECT')")
    A("bpy.ops.object.delete(use_global=False)")
    A("for col in list(bpy.data.collections): bpy.data.collections.remove(col)")
    A("")
    A("# Ground plane")
    A("bpy.ops.mesh.primitive_plane_add(size=300, location=(50,50,0))")
    A("bpy.context.active_object.name = 'Ground'")
    A("")
    A("# --------------- collections ---------------")
    A("rigged_col = bpy.data.collections.new('rigged_cameras')")
    A("app_col    = bpy.data.collections.new('app_cameras')")
    A("bpy.context.scene.collection.children.link(rigged_col)")
    A("bpy.context.scene.collection.children.link(app_col)")
    A("")
    A("FRAMES = " + json.dumps(data, indent=2))
    A("")
    A("# --------------- rigged cameras ---------------")
    A("# Hierarchy: [frame_num] -> yaw.[n] -> pitch.[n] -> Camera.[n]")
    A("# Axes follow your spec:")
    A("#   [frame_num] : translated to GPS position")
    A("#   yaw.[n]     : rotateZ = -heading  (CW heading -> CCW Blender Z)")
    A("#   pitch.[n]   : rotateX = -pitch    (negative pitch = look down)")
    A("#   Camera.[n]  : rotateX=90 base, rotateY=roll")
    A("")
    A("for f in FRAMES:")
    A("    n   = f['frame_num']")
    A("    img = f['img_path']")
    A("")
    A("    top = new_empty(n, loc=(f['x'], f['y'], f['z']))")
    A("")
    A("    yaw = new_empty('yaw.' + n, parent=top)")
    A("    yaw.rotation_mode  = 'XYZ'")
    A("    yaw.rotation_euler = (0, 0, math.radians(-f['heading']))")
    A("")
    A("    pit = new_empty('pitch.' + n, parent=yaw)")
    A("    pit.rotation_mode  = 'XYZ'")
    A("    pit.rotation_euler = (math.radians(f['pitch']), 0, 0)")
    A("")
    A("    cam = new_camera('Camera.' + n, f['fov_deg'], img, parent=pit)")
    A("    cam.rotation_euler = (math.radians(90), math.radians(f['roll']), 0)")
    A("")
    A("    add_to_collection(top, rigged_col)")
    A("")
    A("# --------------- app cameras ---------------")
    A("# These match exactly what the raycast app uses:")
    A("#   Position: camera world position (ENU)")
    A("#   Rotation: built from the same R_cam_from_world matrix the app uses.")
    A("#             R is stored as a 3x3 list of rows.")
    A("#")
    A("# Blender camera convention vs OpenCV:")
    A("#   OpenCV: X=right, Y=down,  Z=forward")
    A("#   Blender: X=right, Y=up,  Z=backward")
    A("# So we flip Y and Z columns: R_bl = R_cv * diag(1,-1,-1)")
    A("")
    A("for f in FRAMES:")
    A("    n   = f['frame_num']")
    A("    img = f['img_path']")
    A("    R   = f['R']")
    A("")
    A("    cam = new_camera('app_Camera.' + n, f['fov_deg'], img)")
    A("    cam.location = (f['x'], f['y'], f['z'])")
    A("")
    A("    if R is not None:")
    A("        # R is R_cam_from_world (3x3). Camera-to-world = R.T")
    A("        Rct = mathutils.Matrix([[R[0][0],R[1][0],R[2][0]],")
    A("                                [R[0][1],R[1][1],R[2][1]],")
    A("                                [R[0][2],R[1][2],R[2][2]]])")
    A("        # Flip Y and Z to convert OpenCV->Blender convention")
    A("        flip = mathutils.Matrix([[1, 0, 0],[0,-1, 0],[0, 0,-1]])")
    A("        Rbl  = Rct @ flip")
    A("        cam.rotation_mode = 'QUATERNION'")
    A("        cam.rotation_quaternion = Rbl.to_quaternion()")
    A("    else:")
    A("        cam.rotation_mode = 'XYZ'")
    A("        cam.rotation_euler = (math.radians(90), 0, 0)")
    A("")
    A("    add_to_collection(cam, app_col)")
    A("")
    A("print('Done.', len(FRAMES), 'cameras in each group.')")

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('Blender script written to:', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--optimize-pitch', action='store_true',
                        help='Run ground-scatter pitch optimizer before export (slower).')
    parser.add_argument('--height', choices=['agl','avg','tor'], default='agl',
                        help='Camera Z: agl=AGL only (default), avg=average both, tor=takeoff_ref only.')
    parser.add_argument('--enhance', action='store_true',
                        help='Enable CLAHE+unsharp preprocessing before LightGlue (off by default).')
    parser.add_argument('--out_dir',    required=True,
                        help='Directory to save undistorted images and blender_scene.py')
    args = parser.parse_args()

    data = collect_data(args.frames_dir, args.out_dir, optimize_pitch=args.optimize_pitch, enhance=args.enhance, height_mode=args.height)

    print(f"{len(data)} frame(s):")
    for d in data:
        print(f"  {d['frame_num']:>6}  "
              f"pos=({d['x']:6.1f},{d['y']:6.1f},{d['z']:5.1f})  "
              f"hdg={d['heading']:5.1f}  pitch={d['pitch']:6.1f}  "
              f"roll={d['roll']:6.1f}  fov={d['fov_deg']:.1f}")

    out = os.path.join(args.out_dir, 'blender_scene.py')
    generate_blender_script(data, out)
    print('Blender: Scripting -> Open blender_scene.py -> Run Script')
