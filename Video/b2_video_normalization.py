#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — Face-only normalization with MediaPipe FaceMesh (468) → AU-level Graph for GCN/ST-GCN
"""

from __future__ import annotations
import re, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import cv2, numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
import mediapipe as mp

# ==========================
# Config
# ==========================
@dataclass
class Config:
    data_root: Path
    out_root: Path
    target_fps: float = 60.0
    resize_wh: Tuple[int, int] = (256, 256)
    write_preview: bool = False
    temporal_stride: int = 1
    model_complexity: int = 1
    refine_landmarks: bool = True
    min_det_conf: float = 0.5
    min_track_conf: float = 0.5
    detect_every_k: int = 3
    ema_alpha: float = 0.8
    face_pad_ratio: float = 0.25
    start_time: Optional[float] = None  # Thời gian bắt đầu (giây)
    end_time: Optional[float] = None    # Thời gian kết thúc (giây)


mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_fd        = mp.solutions.face_detection

# ==========================
# AU node definition
# ==========================
AU_NODES = [
    "brow_left_inner","brow_left_outer","brow_right_inner","brow_right_outer",
    "eye_left_upper","eye_left_lower","eye_right_upper","eye_right_lower",
    "nose_bridge","nose_alar_left","nose_alar_right",
    "upper_lip","lower_lip","lip_corners","chin_center",
]
FEATURE_NAMES = ["cx","cy","cz","vis","area","aspect","dcx","dcy","darea","daspect"]

# ==========================
# Utility helpers
# ==========================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def parse_subject_and_code(video_path: Path) -> Tuple[str, Optional[str]]:
    sid = video_path.parent.name
    # Tìm mã số sau dấu gạch dưới cuối cùng (ví dụ: P001_213 -> 213)
    m = re.search(r"_(\d+)$", video_path.stem)
    return sid, m.group(1) if m else None

def match_target_fps(orig_fps: float, target: float) -> int:
    return max(1, int(round((orig_fps or target) / target)))

def ema_bbox(prev, curr, alpha=0.8):
    if prev is None: return curr
    px,py,pw,ph = prev; cx,cy,cw,ch = curr
    return (alpha*px+(1-alpha)*cx, alpha*py+(1-alpha)*cy,
            alpha*pw+(1-alpha)*cw, alpha*ph+(1-alpha)*ch)

def expand_square_bbox(x,y,w,h,pad,W,H):
    cx,cy = x+w/2, y+h/2
    side = max(w,h)*(1+pad)
    x1=int(max(0,cx-side/2)); y1=int(max(0,cy-side/2))
    x2=int(min(W,cx+side/2));  y2=int(min(H,cy+side/2))
    side2=min(x2-x1,y2-y1)
    return x1,y1,x1+side2,y1+side2

# ---- Convex + temporal ----
def region_pool_stats(lm_xyzv: np.ndarray, idxs: List[int]):
    if len(idxs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    pts = lm_xyzv[idxs]
    if len(pts) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    cx,cy,cz = pts[:,0].mean(),pts[:,1].mean(),pts[:,2].mean()
    vis = pts[:,3].mean()
    if len(idxs)>=3:
        try: area=float(ConvexHull(pts[:,:2]).volume)
        except: area=0.0
    else: area=0.0
    if len(pts) > 0:
        x0,y0 = pts[:,:2].min(0); x1,y1 = pts[:,:2].max(0)
        aspect=float(max(1e-6,(x1-x0))/max(1e-6,(y1-y0)))
    else:
        aspect = 1.0
    return cx,cy,cz,vis,area,aspect

def temporal_features(X):
    T,_=X.shape; d=np.zeros((T,4),np.float32)   # X=[T,D] T frame, D = 6 
    diff=X[1:,[0,1,4,5]]-X[:-1,[0,1,4,5]]; d[1:]=diff
    return np.concatenate([X,d],1)

# ==========================
# AU adjacency
# ==========================
def build_au_adjacency():
    n=len(AU_NODES); A=np.zeros((n,n),np.float32)
    def link(a,b): i,j=AU_NODES.index(a),AU_NODES.index(b); A[i,j]=A[j,i]=1
    link("brow_left_inner","eye_left_upper"); link("brow_left_outer","eye_left_upper")
    link("brow_right_inner","eye_right_upper");link("brow_right_outer","eye_right_upper")
    link("brow_left_inner","nose_bridge");link("brow_right_inner","nose_bridge")
    link("eye_left_upper","eye_left_lower");link("eye_right_upper","eye_right_lower")
    link("nose_bridge","eye_left_upper");link("nose_bridge","eye_right_upper")
    link("nose_bridge","upper_lip");link("nose_alar_left","upper_lip");link("nose_alar_right","upper_lip")
    link("upper_lip","lower_lip");link("upper_lip","lip_corners");link("lower_lip","lip_corners")
    link("chin_center","lower_lip")
    link("brow_left_outer","eye_left_lower");link("brow_right_outer","eye_right_lower")
    link("nose_alar_left","lip_corners");link("nose_alar_right","lip_corners")
    np.fill_diagonal(A,1)
    return A

# ==========================
# Dynamic ROI via FaceMesh connections
# ==========================
def ids_from_edges(edge_set):
    s=set()
    for a,b in edge_set: s.add(a); s.add(b)
    return sorted(s)

FACEMESH_RIGHT_EYEBROW_NODES = ids_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
FACEMESH_LEFT_EYEBROW_NODES  = ids_from_edges(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
FACEMESH_LIPS_NODES          = ids_from_edges(mp_face_mesh.FACEMESH_LIPS)
FACEMESH_RIGHT_EYE_NODES     = ids_from_edges(mp_face_mesh.FACEMESH_RIGHT_EYE)
FACEMESH_LEFT_EYE_NODES      = ids_from_edges(mp_face_mesh.FACEMESH_LEFT_EYE)
FACEMESH_NOSE_NODES          = ids_from_edges(mp_face_mesh.FACEMESH_NOSE)


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
    ys=np.array([xy[i,1] for i in ids]); med=float(np.median(ys))
    up=[i for i in ids if xy[i,1]<med]; lo=[i for i in ids if xy[i,1]>=med]
    return up,lo

# ==========================
# Core extractor
# ==========================
def extract_facemesh_sequence(video_path: Path, cfg: Config):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video_path}")
    orig_fps=cap.get(cv2.CAP_PROP_FPS) or cfg.target_fps
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Tính toán frame bắt đầu và kết thúc dựa trên thời gian
    start_frame = 0
    end_frame = total_frames
    if cfg.start_time is not None:
        start_frame = int(cfg.start_time * orig_fps)
    if cfg.end_time is not None:
        end_frame = int(cfg.end_time * orig_fps)
    
    # Đảm bảo giá trị hợp lệ
    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(start_frame, min(end_frame, total_frames))
    
    # Di chuyển đến frame bắt đầu
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    stride_pre=match_target_fps(orig_fps,cfg.target_fps)
    W,H=cfg.resize_wh
    seq=[]; preview=[] if cfg.write_preview else None
    smooth_bbox=None; last_det=-999; last_valid=None

    with mp_fd.FaceDetection(model_selection=1,min_detection_confidence=cfg.min_det_conf) as fd, \
         mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=cfg.refine_landmarks,
                               min_detection_confidence=cfg.min_det_conf,min_tracking_confidence=cfg.min_track_conf) as fm:
        pbar=tqdm(total=end_frame-start_frame,desc=video_path.name)
        fidx=start_frame; kept=0
        while fidx < end_frame:
            ret,frame=cap.read()
            if not ret: break
            pbar.update(1)
            if fidx%stride_pre!=0: fidx+=1; continue
            if kept%cfg.temporal_stride!=0: fidx+=1; kept+=1; continue

            frame_small=cv2.resize(frame,(W,H))
            rgb=cv2.cvtColor(frame_small,cv2.COLOR_BGR2RGB)
            Hs,Ws=frame_small.shape[:2]
            run_det=(kept-last_det)>=cfg.detect_every_k or smooth_bbox is None
            if run_det:
                det=fd.process(rgb)
                if det.detections:
                    r=det.detections[0].location_data.relative_bounding_box
                    x,y,w,h=r.xmin*Ws,r.ymin*Hs,r.width*Ws,r.height*Hs
                    smooth_bbox=ema_bbox(smooth_bbox,(x,y,w,h),cfg.ema_alpha)
                    last_det=kept
            if smooth_bbox:
                x,y,w,h=smooth_bbox; x1,y1,x2,y2=expand_square_bbox(x,y,w,h,cfg.face_pad_ratio,Ws,Hs)
                roi=frame_small[y1:y2,x1:x2]
                roi_rgb=cv2.cvtColor(cv2.resize(roi,(W,H)),cv2.COLOR_BGR2RGB) if roi.size>0 else rgb
            else: roi_rgb=rgb

            res=fm.process(roi_rgb)
            au=np.zeros((len(AU_NODES),len(FEATURE_NAMES)),np.float32)

            if res.multi_face_landmarks:
                lm=res.multi_face_landmarks[0].landmark
                lm_xyzv=np.zeros((468,4),np.float32)
                for i,p in enumerate(lm[:468]):
                    lm_xyzv[i,:3]=(p.x,p.y,getattr(p,"z",0.0)); lm_xyzv[i,3]=1.0
                xy=lm_xyzv[:,:2]

                # dynamic ROI
                br_i,br_o=split_inner_outer_by_nose(FACEMESH_RIGHT_EYEBROW_NODES,xy,1)
                bl_i,bl_o=split_inner_outer_by_nose(FACEMESH_LEFT_EYEBROW_NODES,xy,1)
                lip_u,lip_l=split_lip_upper_lower(FACEMESH_LIPS_NODES,xy)
                roi={
                    "brow_left_inner":bl_i,"brow_left_outer":bl_o,
                    "brow_right_inner":br_i,"brow_right_outer":br_o,
                    "eye_left_upper":FACEMESH_LEFT_EYE_NODES,"eye_left_lower":FACEMESH_LEFT_EYE_NODES,
                    "eye_right_upper":FACEMESH_RIGHT_EYE_NODES,"eye_right_lower":FACEMESH_RIGHT_EYE_NODES,
                    "nose_bridge":FACEMESH_NOSE_NODES,
                    "nose_alar_left":[i for i in FACEMESH_NOSE_NODES if xy[i,0]<xy[1,0]],
                    "nose_alar_right":[i for i in FACEMESH_NOSE_NODES if xy[i,0]>=xy[1,0]],
                    "upper_lip":lip_u,"lower_lip":lip_l,
                    "lip_corners":[61,291],
                    "chin_center":[152,148,171,175,377,396],
                }
                base=[]
                for node in AU_NODES:  
                    cx,cy,cz,vis,area,asp=region_pool_stats(lm_xyzv,roi[node])
                    base.append([cx,cy,cz,vis,area,asp])
                au[:,:6]=np.asarray(base,np.float32)
                last_valid=au.copy()
            else:
                if last_valid is not None:
                    au=last_valid.copy(); au[:,3]=0.0
            seq.append(au)

            if preview is not None:
                draw=frame_small.copy()
                if res.multi_face_landmarks:
                    mp_drawing.draw_landmarks(draw,res.multi_face_landmarks[0],
                                              mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1))
                if smooth_bbox:
                    x,y,w,h=smooth_bbox; x1,y1,x2,y2=expand_square_bbox(x,y,w,h,cfg.face_pad_ratio,Ws,Hs)
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),1)
                preview.append(draw)
            fidx+=1; kept+=1
        pbar.close()
    cap.release()

    if not seq: raise RuntimeError(f"No frames: {video_path}")
    X=np.stack(seq,0).astype(np.float32)      # [T,N,F] T số frame, N node. F=10 feature
    T,N,_=X.shape
    for n in range(N):
        base=X[:,n,:6]; X[:,n,:10]=temporal_features(base)
    eff_fps=max(1e-6,orig_fps/match_target_fps(orig_fps,cfg.target_fps))/max(1,cfg.temporal_stride)
    return X[:,:,:10],eff_fps,(np.stack(preview) if preview else None)

# ==========================
# Saving
# ==========================
def save_preview(frames,fps,outp:Path):
    h,w=frames[0].shape[:2]
    vw=cv2.VideoWriter(str(outp),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for f in frames: vw.write(f)
    vw.release()

def build_and_save(video:Path,cfg:Config,adj=None)->Path:
    sid,code=parse_subject_and_code(video); code=code or "unknown"
    seq,fps,preview=extract_facemesh_sequence(video,cfg)
    adj=adj if adj is not None else build_au_adjacency()
    meta={
        "subject_id":sid,"ma_mau":code,"fps":fps,
        "T":int(seq.shape[0]),"N_AU":int(seq.shape[1]),
        "video_path":str(video),"feature_names":FEATURE_NAMES,"au_nodes":AU_NODES,
    }
    out=cfg.out_root/f"{sid}_{code}.npz"; ensure_dir(cfg.out_root)
    np.savez_compressed(out,graph_seq=seq.astype(np.float32),
                        adj=adj.astype(np.float32),
                        meta=np.bytes_(json.dumps(meta)))
    if preview is not None:
        save_preview(preview,max(1.0,fps),cfg.out_root/f"{sid}_{code}_preview.mp4")
    return out

def process_dataset(cfg:Config):
    ensure_dir(cfg.out_root); adj=build_au_adjacency()
    videos=[]
    # First, check for videos directly in the data_root
    for ext in ("*.mp4","*.mov","*.mkv","*.avi"):
        videos+=sorted(cfg.data_root.glob(ext))
    # Then, check in subdirectories
    for p in sorted(cfg.data_root.glob("*/")):
        for ext in ("*.mp4","*.mov","*.mkv","*.avi"):
            videos+=sorted(p.glob(ext))
    if not videos: raise RuntimeError(f"No videos under {cfg.data_root}")
    print(f"Found {len(videos)} videos. Processing…")
    for v in videos:
        try: print("✔",build_and_save(v,cfg,adj))
        except Exception as e: print(f"✖ {v}: {e}")

# ==========================
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Face-only normalization with MediaPipe FaceMesh")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing video files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--target-fps", type=float, default=60.0, help="Target FPS (default: 60.0)")
    parser.add_argument("--resize-wh", type=int, nargs=2, default=[256, 256], help="Resize width height (default: 256 256)")
    parser.add_argument("--write-preview", action="store_true", help="Write preview videos")
    parser.add_argument("--start-time", type=float, default=None, help="Start time in seconds (default: from beginning)")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds (default: to end)")
    args = parser.parse_args()
    
    DATA_ROOT = Path(args.input)
    OUT_ROOT = Path(args.output)
    cfg=Config(DATA_ROOT,OUT_ROOT,target_fps=args.target_fps,resize_wh=tuple(args.resize_wh),
               write_preview=args.write_preview,temporal_stride=1,model_complexity=1,
               refine_landmarks=True,min_det_conf=0.5,min_track_conf=0.5,
               detect_every_k=3,ema_alpha=0.8,face_pad_ratio=0.25,
               start_time=args.start_time,end_time=args.end_time)
    process_dataset(cfg)
