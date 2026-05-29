import argparse, sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def collect_data(frames_dir, out_dir, calculate_orientation=False, manual_only=False,
                 enhance=False, height_mode='tor', feature_matcher_debug=False):
    import config
    import cv2
    from pipeline.frame     import load_frames
    from pipeline.ocr       import extract_telemetry_all
    from pipeline.undistort import undistort_all

    os.makedirs(out_dir, exist_ok=True)

    from pipeline.pose import estimate_poses
    frames = load_frames(frames_dir)
    extract_telemetry_all(frames)
    undistort_all(frames)
    estimate_poses(frames)

    # ── Snapshot raw telemetry BEFORE any solve mutates the frames ────────────
    # These are used for the telemetry camera collection in Blender.
    telemetry = {}
    for f in frames:
        telemetry[f.stem] = dict(
            gps_x       = float(f.position_enu[0]) if f.position_enu is not None else 0.0,
            gps_y       = float(f.position_enu[1]) if f.position_enu is not None else 0.0,
            gps_z       = float(f.alt_takeoff_ref_m or 0.0),   # always tor for telemetry
            gps_heading = float(f.heading_deg      or 0.0),
            gps_roll    = float(f.camera_roll_deg  or 0.0),    # bracket or GeoCalib
        )

    if calculate_orientation or manual_only:
        from pipeline.detect_van    import VanDetector
        from pipeline.feature_matcher import refine_pitches, load_manual_pairs, \
                                             set_enhance, set_debug
        set_enhance(enhance)
        set_debug(feature_matcher_debug)
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

        if manual_only:
            # Skip LightGlue — solve from manual correspondences JSON only.
            manual_pairs = load_manual_pairs(frames)
            print(f'Running manual-only orientation solver ({len(manual_pairs)} ground pairs)...')
            refine_pitches(frames, manual_pairwise_features=manual_pairs)
        else:
            detector   = VanDetector()
            detections = detector.detect_all(frames)
            van_bboxes = {f.stem: bbox for f, bbox in detections.items()}
            print('Running orientation solver (LightGlue + manual)...')
            refine_pitches(frames, van_bboxes=van_bboxes)

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
        roll    = f.camera_roll_deg
        pitch   = f.gimbal_pitch_deg if f.gimbal_pitch_deg is not None else 0.0
        fov_deg = 90.0
        if f.K_undist is not None and f.undistorted is not None:
            fov_deg = math.degrees(2.0 * math.atan(
                f.undistorted.shape[1] / (2.0 * float(f.K_undist[0, 0]))))

        R = f.R.tolist() if f.R is not None else None
        tel = telemetry.get(f.stem, {})

        rows.append(dict(
            frame_num   = frame_num,
            img_path    = os.path.abspath(img_path).replace('\\', '/'),
            img_w       = int(f.undistorted.shape[1]) if f.undistorted is not None else 768,
            img_h       = int(f.undistorted.shape[0]) if f.undistorted is not None else 432,
            # solved / height-corrected values
            x           = round(float(enu[0]), 3),
            y           = round(float(enu[1]), 3),
            z           = round(z, 3),
            heading     = round(float(f.heading_deg or 0), 2),
            pitch       = round(pitch, 2),
            roll        = round(roll, 2),
            fov_deg     = round(fov_deg, 3),
            R           = R,
            # raw telemetry (pre-solve snapshot) for telemetry camera collection
            gps_x       = round(tel.get('gps_x',       0.0), 3),
            gps_y       = round(tel.get('gps_y',       0.0), 3),
            gps_z       = round(tel.get('gps_z',       0.0), 3),
            gps_heading = round(tel.get('gps_heading', 0.0), 2),
            gps_roll    = round(tel.get('gps_roll',    0.0), 2),
        ))
    van_bbox = None
    if calculate_orientation or manual_only:
        import config as _cfg
        van_bbox = compute_van_bbox(frames, _cfg.MANUAL_CORRESPONDENCES_FILE)

    return rows, van_bbox


def compute_van_bbox(frames, json_path):
    """
    Recover 3-D world positions of van feature marks by raycasting each
    2-D pixel through the solved camera to the feature's known z-plane.
    Average across all frames that have the feature marked.

    Uses:
      roof_edge  (is_pair, z=VAN_HEIGHT_M, lateral) — rear top corners
      wheel_axis (is_pair, z=WHEEL_RADIUS_M, forward) — front/rear wheel centres

    Bbox construction (per user spec):
      1. Roof-edge A and B   → lateral direction, van width, roof height
      2. Wheel-axis midpoint → longitudinal+lateral centre at wheel height
      3. Move from wheel_mid along roof-edge direction by half roof-edge length
         → lateral centre of van at wheel height
      4. Raise to mid-height between wheel_z and roof_z → bbox centre
      5. Vector from bbox-centre to roof_A → three half-extents (length, width, height)

    Returns a dict ready to embed in the Blender script, or None if the
    required feature types are absent from the JSON.
    """
    import json as _json, math as _math
    from pathlib import Path
    import numpy as np
    import config

    path = Path(json_path)
    if not path.exists():
        return None

    data = _json.loads(path.read_text(encoding='utf-8'))
    corrs = data.get('correspondences', [])

    roof_feats  = [c for c in corrs if c.get('type') == 'roof_edge']
    wheel_feats = [c for c in corrs if c.get('type') == 'wheel_axis']

    if not roof_feats or not wheel_feats:
        print(f"Van bbox: roof_edge={len(roof_feats)} wheel_axis={len(wheel_feats)} "
              f"feature(s) — skipping bbox.")
        return None

    stem_to_frame = {f.stem: f for f in frames}

    def raycast(frame, u, v, z_plane):
        """Raycast pixel (u,v) in solved frame to z=z_plane → world point or None."""
        if frame.R is None or frame.position_enu is None or frame.K_undist is None:
            return None
        K_inv   = np.linalg.inv(frame.K_undist)
        d_cam   = K_inv @ np.array([u, v, 1.0])
        d_world = frame.R.T @ d_cam          # R is cam-from-world → R.T = world-from-cam
        norm    = float(np.linalg.norm(d_world))
        if norm < 1e-12:
            return None
        d_world /= norm
        dz = float(d_world[2])
        if abs(dz) < 1e-9:
            return None
        t = (z_plane - float(frame.position_enu[2])) / dz
        if t <= 0:
            return None
        return frame.position_enu + t * d_world

    def pair_3d(feats):
        """Average 3-D positions of A and B endpoints across all frames."""
        pts_a, pts_b = [], []
        for feat in feats:
            z = float(feat.get('z_plane', 0.0))
            for stem, pts in feat.get('points', {}).items():
                fr = stem_to_frame.get(stem)
                if fr is None or len(pts) < 2:
                    continue
                Pa = raycast(fr, pts[0][0], pts[0][1], z)
                Pb = raycast(fr, pts[1][0], pts[1][1], z)
                if Pa is not None and Pb is not None:
                    pts_a.append(Pa)
                    pts_b.append(Pb)
        if not pts_a:
            return None, None
        return np.mean(pts_a, axis=0), np.mean(pts_b, axis=0)

    roof_A, roof_B   = pair_3d(roof_feats)
    wheel_A, wheel_B = pair_3d(wheel_feats)

    if roof_A is None or wheel_A is None:
        print("Van bbox: could not back-project feature marks — skipping bbox.")
        return None

    # ── Lateral direction from roof back edge ────────────────────────────────
    roof_vec   = roof_B - roof_A                          # lateral vector
    roof_len   = float(np.linalg.norm(roof_vec[:2]))      # horizontal length (≈ VAN_WIDTH_M)
    if roof_len < 1e-6:
        print("Van bbox: roof_edge endpoints coincide — skipping bbox.")
        return None
    roof_dir3d = roof_vec / float(np.linalg.norm(roof_vec))   # full 3-D unit vector

    # ── Wheelbase midpoint → lateral centre → vertical centre ────────────────
    wheel_mid  = (wheel_A + wheel_B) / 2.0
    lat_centre = wheel_mid + roof_dir3d * (roof_len / 2.0)   # lateral centre, wheel height
    roof_z     = float((roof_A[2]  + roof_B[2])  / 2.0)
    wheel_z    = float((wheel_A[2] + wheel_B[2]) / 2.0)
    ground_z   = wheel_z - config.WHEEL_RADIUS_M   # true ground under the van
    bbox_cz    = (ground_z + roof_z) / 2.0
    bbox_centre = np.array([lat_centre[0], lat_centre[1], bbox_cz])

    # ── Half-extents from bbox-centre → roof_A ───────────────────────────────
    # Van forward axis = wheelbase direction (A→B along the side of the van)
    wb_vec = (wheel_B - wheel_A)[:2]
    wb_len = float(np.linalg.norm(wb_vec))
    if wb_len < 1e-6:
        print("Van bbox: wheel_axis endpoints coincide — skipping bbox.")
        return None
    wb_unit  = wb_vec / wb_len                                     # 2-D forward unit
    van_hdg  = _math.degrees(_math.atan2(float(wb_unit[0]),
                                          float(wb_unit[1]))) % 360.0
    van_fwd  = np.array([float(wb_unit[0]), float(wb_unit[1]), 0.0])
    van_lat  = np.array([float(wb_unit[1]), -float(wb_unit[0]), 0.0])  # 90° CW

    delta       = roof_A - bbox_centre
    half_length = abs(float(np.dot(delta, van_fwd)))
    half_width  = abs(float(np.dot(delta, van_lat)))
    half_height = abs(float(delta[2]))

    result = dict(
        center_x    = round(float(bbox_centre[0]), 3),
        center_y    = round(float(bbox_centre[1]), 3),
        center_z    = round(float(bbox_centre[2]), 3),
        size_length = round(2 * half_length, 3),
        size_lateral= round(2 * half_width,  3),
        size_height = round(2 * half_height, 3),
        heading_deg = round(van_hdg, 2),
        # Raw 3-D positions of each feature mark — used for locators in Blender.
        locators    = [],
    )

    def _add_locators(feats, label):
        for feat_i, feat in enumerate(feats):
            z = float(feat.get('z_plane', 0.0))
            is_pair = feat.get('is_pair', False)
            for stem, pts in feat.get('points', {}).items():
                fr = stem_to_frame.get(stem)
                if fr is None:
                    continue
                if is_pair and len(pts) >= 2:
                    for end, pt in zip(('A', 'B'), pts):
                        P = raycast(fr, pt[0], pt[1], z)
                        if P is not None:
                            result['locators'].append(dict(
                                name  = f"{label}_{feat_i}_{end}_{stem[-8:]}",
                                x=round(float(P[0]),3), y=round(float(P[1]),3), z=round(float(P[2]),3),
                                color = (1.0, 0.4, 0.1) if label=='roof' else (0.2, 0.8, 1.0),
                            ))
                elif not is_pair:
                    P = raycast(fr, pts[0], pts[1], z)
                    if P is not None:
                        result['locators'].append(dict(
                            name  = f"{label}_{feat_i}_{stem[-8:]}",
                            x=round(float(P[0]),3), y=round(float(P[1]),3), z=round(float(P[2]),3),
                            color = (1.0, 0.4, 0.1) if label=='roof' else (0.2, 0.8, 1.0),
                        ))

    _add_locators(roof_feats,  'roof_edge')
    _add_locators(wheel_feats, 'wheel_axis')

    print(
        f"Van bbox: centre=({result['center_x']:.2f}, {result['center_y']:.2f}, "
        f"{result['center_z']:.2f})m  "
        f"dims={result['size_length']:.2f}×{result['size_lateral']:.2f}×{result['size_height']:.2f}m  "
        f"hdg={result['heading_deg']:.1f}°  {len(result['locators'])} locator(s)"
    )
    return result


def generate_blender_script(data, out_path, van_bbox=None):
    lines = []
    A = lines.append

    # Bake the undistorted image resolution into the script so Blender's
    # camera frustums match the actual pixels.
    if data:
        img_w = data[0].get('img_w', 768)
        img_h = data[0].get('img_h', 432)
    else:
        img_w, img_h = 768, 432

    A("import bpy, math, mathutils, os")
    A("")
    A(f"# Render resolution — must match undistorted images exactly")
    A(f"bpy.context.scene.render.resolution_x = {img_w}")
    A(f"bpy.context.scene.render.resolution_y = {img_h}")
    A(f"bpy.context.scene.render.resolution_percentage = 100")
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
    A("def new_locator(name, loc, display_type='PLAIN_AXES', size=0.3):")
    A("    o = bpy.data.objects.new(name, None)")
    A("    o.empty_display_type = display_type")
    A("    o.empty_display_size = size")
    A("    bpy.context.scene.collection.objects.link(o)")
    A("    o.location = loc")
    A("    return o")
    A("")
    A("def new_camera(name, fov_deg, img_path, parent=None):")
    A("    cd = bpy.data.cameras.new(name)")
    A("    cd.lens_unit   = 'FOV'")
    A("    cd.angle       = math.radians(fov_deg)   # horizontal FOV")
    A("    cd.sensor_fit  = 'HORIZONTAL'             # lock angle to horizontal axis")
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
    A("    for child in obj.children_recursive:")
    A("        if child.name not in col.objects:")
    A("            col.objects.link(child)")
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
    A("rigged_col    = bpy.data.collections.new('rigged_cameras')")
    A("app_col       = bpy.data.collections.new('app_cameras')")
    A("telemetry_col = bpy.data.collections.new('telemetry_cameras')")
    A("bpy.context.scene.collection.children.link(rigged_col)")
    A("bpy.context.scene.collection.children.link(app_col)")
    A("bpy.context.scene.collection.children.link(telemetry_col)")
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
    A("")
    A("# --------------- telemetry cameras ---------------")
    A("# Raw telemetry — no solver corrections applied.")
    A("# Position : GPS ENU (x, y) + Z from takeoff-reference barometer (tor).")
    A("# Yaw      : OCR compass heading.")
    A("# Pitch    : 0  (gimbal pitch ignored — pure platform attitude).")
    A("# Roll     : horizon bracket detector (or GeoCalib fallback).")
    A("# Same rig hierarchy as rigged_cameras so the two can be compared directly.")
    A("")
    A("for f in FRAMES:")
    A("    n   = f['frame_num']")
    A("    img = f['img_path']")
    A("")
    A("    top = new_empty('tel.' + n, loc=(f['gps_x'], f['gps_y'], f['gps_z']))")
    A("")
    A("    yaw = new_empty('tel.yaw.' + n, parent=top)")
    A("    yaw.rotation_mode  = 'XYZ'")
    A("    yaw.rotation_euler = (0, 0, math.radians(-f['gps_heading']))")
    A("")
    A("    pit = new_empty('tel.pitch.' + n, parent=yaw)")
    A("    pit.rotation_mode  = 'XYZ'")
    A("    pit.rotation_euler = (0, 0, 0)  # pitch = 0")
    A("")
    A("    cam = new_camera('tel.Camera.' + n, f['fov_deg'], img, parent=pit)")
    A("    cam.rotation_euler = (math.radians(90), math.radians(f['gps_roll']), 0)")
    A("")
    A("    add_to_collection(top, telemetry_col)")
    A("")
    A("# --------------- van bounding box ---------------")
    A("# Wireframe box derived from roof_edge and wheel_axis manual marks.")
    A("# Position and heading solved from the final camera calibration.")
    A("# Local axes: X=lateral(right), Y=forward, Z=up  — rotation around Z = -heading.")
    A(f"VAN_BBOX = {json.dumps(van_bbox)}")
    A("")
    A("if VAN_BBOX:")
    A("    b = VAN_BBOX")
    A("    bpy.ops.mesh.primitive_cube_add(size=1,")
    A("        location=(b['center_x'], b['center_y'], b['center_z']))")
    A("    van_box = bpy.context.active_object")
    A("    van_box.name = 'Van_BBox'")
    A("    van_box.scale = (b['size_lateral'], b['size_length'], b['size_height'])")
    A("    van_box.rotation_mode = 'XYZ'")
    A("    van_box.rotation_euler = (0, 0, math.radians(-b['heading_deg']))")
    A("    van_box.display_type = 'WIRE'")
    A("    mat = bpy.data.materials.new('VanBBox_Mat')")
    A("    mat.use_nodes = True")
    A("    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (1, 0.4, 0, 1)")
    A("    van_box.data.materials.append(mat)")
    A("    print(f\"Van bbox: {b['size_length']:.2f} × {b['size_lateral']:.2f} × {b['size_height']:.2f} m  hdg={b['heading_deg']:.1f}°\")")
    A("")
    A("    # Feature locators — one plain-axis empty per 2D mark, placed at the")
    A("    # 3-D world position recovered by ray-plane intersection.")
    A("    # Orange = roof_edge endpoints.  Cyan = wheel_axis endpoints.")
    A("    # Each locator is named: type_featIndex_endpoint_frameStem")
    A("    for loc in b.get('locators', []):")
    A("        o = new_locator(loc['name'], (loc['x'], loc['y'], loc['z']))")
    A("        mat_l = bpy.data.materials.new(loc['name'] + '_mat')")
    A("        mat_l.use_nodes = True")
    A("        r, g, bl = loc['color']")
    A("        mat_l.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (r, g, bl, 1)")
    A("        # Can't assign material to empty — use object color instead")
    A("        o.color = (r, g, bl, 1.0)")
    A("")
    A("print('Done.', len(FRAMES), 'cameras in each group.')")

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print('Blender script written to:', out_path)


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level  = logging.INFO,
        format = '%(message)s',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--calculate-orientation', action='store_true',
                        dest='calculate_orientation',
                        help='Run full orientation solver (LightGlue + manual correspondences) before export.')
    parser.add_argument('--manual-only', action='store_true',
                        dest='manual_only',
                        help='Run orientation solver using manual correspondences JSON only (no LightGlue). '
                             'Faster and deterministic; requires a populated manual_correspondences.json.')
    parser.add_argument('--height', choices=['agl','avg','tor'], default='tor',
                        help='Camera Z: tor=takeoff_ref (default), agl=AGL only, avg=average both.')
    parser.add_argument('--enhance', action='store_true',
                        help='Enable CLAHE+unsharp preprocessing before LightGlue (off by default).')
    parser.add_argument('--feature-matcher-debug', action='store_true',
                        dest='feature_matcher_debug',
                        help='Save annotated match images to {out_dir}/debug/ for every matched frame pair.')
    parser.add_argument('--out_dir',    required=True,
                        help='Directory to save undistorted images and blender_scene.py')
    args = parser.parse_args()

    data, van_bbox = collect_data(args.frames_dir, args.out_dir,
                        calculate_orientation=args.calculate_orientation,
                        manual_only=args.manual_only,
                        enhance=args.enhance, height_mode=args.height,
                        feature_matcher_debug=args.feature_matcher_debug)

    print(f"{len(data)} frame(s):")
    for d in data:
        print(f"  {d['frame_num']:>6}  "
              f"pos=({d['x']:6.1f},{d['y']:6.1f},{d['z']:5.1f})  "
              f"hdg={d['heading']:5.1f}  pitch={d['pitch']:6.1f}  "
              f"roll={d['roll']:6.1f}  fov={d['fov_deg']:.1f}")

    out = os.path.join(args.out_dir, 'blender_scene.py')
    generate_blender_script(data, out, van_bbox=van_bbox)
    print('Blender: Scripting -> Open blender_scene.py -> Run Script')
