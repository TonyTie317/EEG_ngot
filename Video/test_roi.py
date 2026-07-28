#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Debug Visualizer for MediaPipe FaceMesh (AU-level regions, dynamic per-frame)
- Brows: split inner/outer by distance to nose tip (id=1)
- Lips: split upper/lower by median-y within FACEMESH_LIPS
- Nose alar: left/right by x relative to nose tip
- Eyes: use full eye node sets (you can specialize to lids if needed)
- Chin: subset of face oval (fixed small set)
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
from itertools import cycle

# =========================
# Helpers for dynamic ROIs
# =========================

def norm2pix(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

def points_from_indices(landmarks, idxs, w, h):
    pts = []
    for i in idxs:
        if 0 <= i < len(landmarks):
            x, y = norm2pix(landmarks[i], w, h)
            pts.append([x, y])
    return np.array(pts, dtype=np.int32) if len(pts) > 0 else None

def draw_filled_polygon(img, pts, color, alpha=0.35, border_thickness=2):
    if pts is None or len(pts) < 3:
        return
    hull = cv2.convexHull(pts)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, hull, color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [hull], isClosed=True, color=color, thickness=border_thickness)

def put_label(img, pts, text, color):
    if pts is None or len(pts) == 0:
        return
    c = pts.mean(axis=0).astype(int)
    x, y = int(c[0]), int(c[1])
    cv2.putText(img, text, (x - 20, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x - 20, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def is_int_string(s):
    try:
        int(s); return True
    except:
        return False

def ids_from_edges(edge_set):
    """Collect unique node ids from a set of edges."""
    s = set()
    for a, b in edge_set:
        s.add(a); s.add(b)
    return sorted(s)

def split_inner_outer_by_nose(ids, xy, nose_id=6):
    """Split a brow set into inner/outer halves by distance to the nose tip."""
    if len(ids) == 0: return [], []
    pts = np.array([xy[i] for i in ids], dtype=np.float32)  # [n,2]
    nose = xy[nose_id]
    d = np.linalg.norm(pts - nose, axis=1)
    order = np.argsort(d)
    k = max(1, len(ids) // 2)
    inner = [ids[i] for i in order[:k]]
    outer = [ids[i] for i in order[k:]]
    return inner, outer

def split_lip_upper_lower(ids, xy):
    """Split lips into upper/lower by median y within the lip node set."""
    if len(ids) == 0: return [], []
    ys = np.array([xy[i, 1] for i in ids], dtype=np.float32)
    med = float(np.median(ys))
    upper = [i for i in ids if xy[i,1] < med]
    lower = [i for i in ids if xy[i,1] >= med]
    return upper, lower

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="0/1/... cho webcam, hoặc đường dẫn video")
    parser.add_argument("--mirror", type=int, default=1, help="1 = hiển thị mirror (selfie), 0 = không")
    parser.add_argument("--draw_mesh", type=int, default=0, help="1 = vẽ full FaceMesh, 0 = tắt")
    parser.add_argument("--alpha", type=float, default=0.35, help="Độ trong suốt fill ROI (0-1)")
    parser.add_argument("--thickness", type=int, default=2, help="Độ dày viền ROI")
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.source)) if is_int_string(args.source) else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Không mở được nguồn video:", args.source)
        return

    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # Precompute node id sets from connections
    EYEBROW_R = ids_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
    EYEBROW_L = ids_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
    LIPS      = ids_from_edges(mp_face_mesh.FACEMESH_LIPS)
    EYE_R     = ids_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
    EYE_L     = ids_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
    NOSE      = ids_from_edges(mp_face_mesh.FACEMESH_NOSE)
    # Chin: lấy vài điểm cằm từ mặt oval (giữ tĩnh một nhóm nhỏ ổn định)
    CHIN_FIXED = [152,148,171,175,377,396]

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    base_palette = [
        (255, 64, 64), (64, 255, 64), (64, 64, 255),
        (255, 200, 0), (0, 200, 255), (200, 0, 255),
        (180, 180, 60), (60, 180, 180), (180, 60, 180),
        (120, 120, 255), (255, 120, 120), (120, 255, 120),
        (255, 170, 40), (40, 170, 255), (170, 40, 255),
    ]
    color_cycle = cycle(base_palette)

    draw_mesh = bool(args.draw_mesh)
    mirror_on = bool(args.mirror)
    print("[Hướng dẫn] Phím: m=toggle mirror, l=toggle mesh, q/ESC=thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        disp = cv2.flip(frame, 1) if mirror_on else frame.copy()
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        h, w = disp.shape[:2]

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark

            if draw_mesh:
                mp_draw.draw_landmarks(
                    image=disp,
                    landmark_list=result.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                )

            # ---- Build dynamic ROIs per frame ----
            # Collect XY normalized in image (not cropped; ok for visualization)
            xy = np.array([[p.x, p.y] for p in lms], dtype=np.float32)  # [468,2]
            nose_tip_id = 1
            # Brows
            brow_r_in, brow_r_out = split_inner_outer_by_nose(EYEBROW_R, xy, nose_tip_id)
            brow_l_in, brow_l_out = split_inner_outer_by_nose(EYEBROW_L, xy, nose_tip_id)
            # Lips
            lip_upper, lip_lower = split_lip_upper_lower(LIPS, xy)
            # Nose alar L/R by x relative to tip
            nose_left  = [i for i in NOSE if xy[i,0] < xy[nose_tip_id,0]]
            nose_right = [i for i in NOSE if xy[i,0] >= xy[nose_tip_id,0]]

            # AU-style ROI dictionary for drawing
            ROI_DYNAMIC = {
                "brow_left_inner":  brow_l_in,
                "brow_left_outer":  brow_l_out,
                "brow_right_inner": brow_r_in,
                "brow_right_outer": brow_r_out,
                "eye_left":         EYE_L,
                "eye_right":        EYE_R,
                "nose_bridge":      NOSE,          # toàn mũi (bạn có thể rút gọn sống mũi nếu muốn)
                "nose_alar_left":   nose_left,
                "nose_alar_right":  nose_right,
                "upper_lip":        lip_upper,
                "lower_lip":        lip_lower,
                "lip_corners":      [61, 291],
                "chin_center":      CHIN_FIXED,
            }

            # ---- Draw all ROIs ----
            for name, idxs in ROI_DYNAMIC.items():
                color = next(color_cycle)
                pts = points_from_indices(lms, idxs, w, h)
                draw_filled_polygon(disp, pts, color, alpha=args.alpha, border_thickness=args.thickness)
                put_label(disp, pts, name, color)

            # reset vòng màu (để màu ổn định theo lượt vẽ)
            color_cycle = cycle(base_palette)

        hud = ("[m]irror:%s  [l]andmarks:%s  [q]uit"
               % ("ON" if mirror_on else "OFF", "ON" if draw_mesh else "OFF"))
        cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("ROI Debug Visualizer (AU-level, dynamic)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('m'):
            mirror_on = not mirror_on
        elif key == ord('l'):
            draw_mesh = not draw_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
