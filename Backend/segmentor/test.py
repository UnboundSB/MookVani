"""YAP Phase-A webcam segmentation debugger (53-point corpus layout).

Place in Backend/segmentor/yap_live_debug.py. Requires your project's
segmentor/yap_conformer.py and trained Phase A checkpoint.

Install: pip install opencv-python mediapipe torch numpy
Run: python segmentor/yap_live_debug.py --download-models
     python segmentor/yap_live_debug.py
Controls: Q quit | R reset | S screenshot | D debug panel | M mirror display

Displays raw MediaPipe landmarks (not cleaned predictions), the segmentation
scores, and actual SegmentTracker completions. NO WORD CLASSIFICATION.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
# Handles execution from Backend/segmentor or another location.
for candidate in [BASE, BASE / 'segmentor', BASE.parent / 'segmentor', Path.cwd() / 'segmentor']:
    if (candidate / 'yap_conformer.py').is_file():
        sys.path.insert(0, str(candidate))
        PROJECT_SEGMENTOR = candidate
        break
else:
    raise SystemExit('Cannot locate segmentor/yap_conformer.py. Put this script in Backend/segmentor/.')
from yap_conformer import YAP, YAPConfig, CORPUS_UPPER_BODY_LANDMARKS, CORPUS_FACE_LANDMARKS

# Official MediaPipe Tasks models. Only downloaded on request.
MODELS = {
    'hand': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'pose': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'face': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
}
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
              (13,17),(17,18),(18,19),(19,20),(0,17)]
COLORS = {'left': (255, 225, 60), 'right': (70, 220, 255), 'body': (130, 255, 110), 'face': (255, 125, 235)}


def args_parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, default=PROJECT_SEGMENTOR/'models/yap_stage2/yap_phaseA_best.pth')
    p.add_argument('--models-dir', type=Path, default=PROJECT_SEGMENTOR/'models/mediapipe_tasks')
    p.add_argument('--download-models', action='store_true', help='Download official MediaPipe Tasks assets if missing')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--width', type=int, default=960)
    p.add_argument('--height', type=int, default=540)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threshold-on', type=float, default=None, help='Override presence on threshold for diagnosis')
    p.add_argument('--threshold-cut', type=float, default=None, help='Override cut threshold for diagnosis')
    p.add_argument('--save-dir', type=Path, default=Path('yap_live_capture'))
    return p.parse_args()


def load_yap(path, device):
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {path}. Train Phase A or provide --checkpoint.')
    # Checkpoint is local/trusted training output. Try safe loading first.
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Could not safely load checkpoint {path}: {exc}') from exc
    state = obj.get('yap', obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict) or not any(k.startswith('core.') for k in state):
        raise ValueError('Expected a Phase A checkpoint with YAP core.* state_dict keys.')
    # Explicitly check model dimensions to avoid silent mismatches.
    in_weight = state['core.lift.0.weight']
    d_model = in_weight.shape[0]
    n_mem = len({key.split('.')[2] for key in state if key.startswith('core.layers.') and key.split('.')[2].isdigit()})
    cnn_kernel = state['core.stem.dw'].shape[0]
    summary_channels = state['core.summary_head.weight'].shape[0]
    entity_dim = state['readout.0.weight'].shape[0]
    hippo_order = state['readout_norm.weight'].numel() // summary_channels
    cfg = YAPConfig.corpus(d_model=d_model, cnn_kernel=cnn_kernel, n_mem_layers=n_mem,
                           summary_channels=summary_channels, hippo_order=hippo_order,
                           entity_dim=entity_dim)
    model = YAP(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, obj


def make_landmarkers(model_dir):
    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError('Install MediaPipe: pip install mediapipe') from e
    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    def mk(kind):
        model_path = str(model_dir / (kind + '.task'))
        common = dict(base_options=BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO)
        if kind == 'hand':
            return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**common, num_hands=2,
                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        if kind == 'pose':
            return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(**common,
                    num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(**common, num_faces=1,
                    min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
    return mp, mk('hand'), mk('pose'), mk('face')


def coords(landmarks, indices=None):
    pts = landmarks if indices is None else [landmarks[i] for i in indices]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def extract_parts(hands, pose_result, face_result, width):
    """Return (53,3), (53,), (53,), per-part raw points for visualisation.

    NOTE hand label mapping is mirrored-camera dependent. MediaPipe handedness
    assumes selfie-mirrored imagery. We flip input BEFORE detection so its
    Left/Right labels match model part order. Verify on your own camera.
    """
    xyz = np.zeros((53, 3), np.float32)
    conf = np.zeros(53, np.float32)
    valid = np.zeros(53, dtype=np.bool_)
    parts = {}
    found_hands = set()
    for pts, handed in zip(hands.hand_landmarks, hands.handedness):
        if not handed:
            continue
        side = handed[0].category_name.lower()
        if side not in ('left', 'right') or side in found_hands:
            continue
        found_hands.add(side)
        st = 0 if side == 'left' else 21
        xyz[st:st+21] = coords(pts)
        # MediaPipe task doesn't expose per-joint detection probability for hands.
        # Presence is binary for detected hand; do NOT treat hand score as per-joint confidence.
        conf[st:st+21] = 1.0
        valid[st:st+21] = True
        parts[side] = xyz[st:st+21].copy()
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        bp = coords(landmarks, CORPUS_UPPER_BODY_LANDMARKS)
        visibility = np.array([getattr(landmarks[i], 'visibility', 1.0) or 0.0
                               for i in CORPUS_UPPER_BODY_LANDMARKS], np.float32)
        xyz[42:46] = bp
        conf[42:46] = visibility
        valid[42:46] = np.isfinite(bp).all(axis=1) & (visibility > 0.15)
        parts['body'] = bp
    if face_result.face_landmarks:
        fp = coords(face_result.face_landmarks[0], CORPUS_FACE_LANDMARKS)
        xyz[46:53] = fp
        conf[46:53] = 1.0
        valid[46:53] = np.isfinite(fp).all(axis=1)
        parts['face'] = fp
    valid &= np.isfinite(xyz).all(axis=1)
    xyz[~valid] = 0.0
    conf[~valid] = 0.0
    return xyz, conf, valid, parts


def point(pt, w, h):
    return int(np.clip(pt[0] * w, 0, w-1)), int(np.clip(pt[1] * h, 0, h-1))


def draw_landmarks(frame, parts):
    h, w = frame.shape[:2]
    for side in ('left','right'):
        if side not in parts: continue
        pts, color = parts[side], COLORS[side]
        for i,j in HAND_EDGES:
            cv2.line(frame, point(pts[i],w,h), point(pts[j],w,h), color, 2, cv2.LINE_AA)
        for i,pt in enumerate(pts):
            cv2.circle(frame, point(pt,w,h), 5 if i in (0,4,8,12,16,20) else 3, color, -1, cv2.LINE_AA)
        x,y = point(pts[0],w,h)
        cv2.putText(frame, f'{side.upper()} HAND', (x+8,max(20,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if 'body' in parts:
        pts = parts['body']; color=COLORS['body']
        for i,j in ((0,1),(0,2),(1,3)):
            cv2.line(frame,point(pts[i],w,h),point(pts[j],w,h),color,2,cv2.LINE_AA)
        for p in pts: cv2.circle(frame,point(p,w,h),5,color,-1)
    if 'face' in parts:
        for i,p in enumerate(parts['face']):
            x,y=point(p,w,h)
            cv2.circle(frame,(x,y),4,COLORS['face'],-1)
            cv2.putText(frame,str(i+46),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.34,COLORS['face'],1)


def text(frame,s,x,y,color=(235,235,235),size=0.52,thick=1):
    cv2.putText(frame,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,thick,cv2.LINE_AA)


def chart(frame, history, x,y,w,h, label, color, threshold):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(35,37,45),-1)
    ty = y+h-int(threshold*h)
    cv2.line(frame,(x,ty),(x+w,ty),(80,85,90),1)
    if len(history)>1:
        arr=list(history)
        coords=[(x+int(i*(w-1)/max(1,len(arr)-1)), y+h-int(np.clip(v,0,1)*h)) for i,v in enumerate(arr)]
        cv2.polylines(frame,[np.array(coords,np.int32)],False,color,2,cv2.LINE_AA)
    text(frame,label,x+6,y+17,color,0.45,1)
    text(frame,f'{history[-1]:.3f}' if history else '---', x+w-75,y+17,(255,255,255),0.45,1)


def overlay(frame, stats, pres_hist, end_hist, events, cfg, debug=True):
    h,w=frame.shape[:2]
    panel_w=400
    canvas=np.zeros((max(h,540), w+panel_w,3),np.uint8)
    canvas[:h,:w]=frame
    canvas[:,w:]=(25,27,33)
    x=w+15
    text(canvas,'YAP | LIVE SEGMENTATION',x,33,(245,245,245),0.67,2)
    text(canvas,'Phase A: boundaries only (NOT words)',x,56,(160,180,195),0.43)
    is_active=stats.get('active',False)
    status='SEGMENT ACTIVE' if is_active else 'WAITING / IDLE'
    state_color=(45,205,130) if is_active else (140,150,160)
    cv2.rectangle(canvas,(x,70),(x+365,105),(35,65,45) if is_active else (48,48,48),-1)
    text(canvas,status,x+12,94,state_color,0.58,2)
    chart(canvas,pres_hist,x,122,365,82,'GESTURE PRESENCE', (70,220,130),cfg.on_thr)
    chart(canvas,end_hist,x,215,365,82,'END / CUT PROBABILITY', (80,165,255),cfg.cut_thr)
    text(canvas,f'Hand tracked: {stats.get("hands",0)}/2 | joints: {stats.get("valid",0)}/53',x,325,(240,240,240),0.50)
    text(canvas,f'Model hand_ok: {stats.get("ok",False)} | speed: {stats.get("speed",0):.3f}',x,348,(185,195,205),0.49)
    text(canvas,f'Completed segments: {stats.get("count",0)}',x,375,(90,230,170),0.55,2)
    text(canvas,f'Processing FPS: {stats.get("fps",0):.1f}',x,402,(180,200,215),0.5)
    text(canvas,'LATEST EVENTS',x,432,(225,220,150),0.5,2)
    for j,event in enumerate(list(events)[-3:][::-1]):
        text(canvas,event[:53],x,454+j*23,(215,215,220),0.41)
    text(canvas,'Q quit | R reset | S screenshot',x,520,(145,160,170),0.43)
    text(canvas,'D panel | M mirror view',x,540,(145,160,170),0.43)
    if stats.get('valid',0)==0:
        text(canvas,'NO VALID LANDMARKS - model receives invalid masks',25,35,(20,30,240),0.63,2)
    elif stats.get('hands',0)==0:
        text(canvas,'NO HANDS DETECTED - check camera / lighting',25,35,(25,160,255),0.58,2)
    if stats.get('just_cut'):
        cv2.rectangle(canvas,(2,2),(w-3,h-3),(80,240,80),6)
        text(canvas,'SEGMENT COMPLETED',25,h-28,(75,245,100),0.8,2)
    if not debug: return canvas[:h,:w]
    return canvas


def main():
    args=args_parse()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    #DEVICE SEUP+===========================================
    device = 'cpu' if (args.device=='cuda' or args.device=='auto' and torch.cuda.is_available()) else 'cpu'
    try:
        model, checkpoint = load_yap(args.checkpoint,device)
    except Exception as exc:
        raise SystemExit(f'[CHECKPOINT ERROR] {exc}') from exc
    cfg=model.cfg
    if args.threshold_on is not None: cfg.on_thr=args.threshold_on
    if args.threshold_cut is not None: cfg.cut_thr=args.threshold_cut
    print(f'[YAP] checkpoint: {args.checkpoint} | device: {device} | J={cfg.total_joints}')
    print(f'[YAP] on_thr={cfg.on_thr} off_thr={cfg.off_thr} cut_thr={cfg.cut_thr}')
    if isinstance(checkpoint,dict): print(f'[YAP] epoch={checkpoint.get("epoch", "unknown")}')
    args.models_dir.mkdir(parents=True,exist_ok=True)
    missing=[]
    for kind,url in MODELS.items():
        dest=args.models_dir/f'{kind}.task'
        if not dest.is_file() or dest.stat().st_size==0:
            if args.download_models:
                print(f'[DOWNLOAD] {kind}: {url}')
                urllib.request.urlretrieve(url,dest)
            else: missing.append(str(dest))
    if missing:
        raise SystemExit('MediaPipe assets missing:\n  '+'\n  '.join(missing)+'\nRun once with --download-models')
    mp,hand,body,face=make_landmarkers(args.models_dir)
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY)
    if not cap.isOpened(): raise SystemExit(f'Cannot open camera {args.camera}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    stream=model.stream(device=device)
    start=time.perf_counter()
    last_ts=-1
    hist_p=collections.deque(maxlen=180);hist_q=collections.deque(maxlen=180)
    events=collections.deque(maxlen=50)
    count=0; show_panel=True; mirror=True; flash_until=0; fps=0.0
    args.save_dir.mkdir(parents=True,exist_ok=True)
    with hand,body,face,torch.inference_mode():
        try:
            while True:
                loop_start=time.perf_counter()
                ok,frame=cap.read()
                if not ok: print('[WARN] Camera frame failed');break
                # ALWAYS run inference in selfie-mirrored coordinates to match MP handedness.
                mirrored=cv2.flip(frame,1)
                rgb=cv2.cvtColor(mirrored,cv2.COLOR_BGR2RGB)
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                ts=max(last_ts+1,int((time.perf_counter()-start)*1000))
                last_ts=ts
                hands=hand.detect_for_video(image,ts)
                poses=body.detect_for_video(image,ts)
                faces=face.detect_for_video(image,ts)
                xyz,confidence,valid,parts=extract_parts(hands,poses,faces,mirrored.shape[1])
                entities=stream.push(torch.from_numpy(xyz),torch.from_numpy(confidence),
                                     torch.from_numpy(valid),ts/1000.0)
                current=time.perf_counter()
                if entities:
                    flash_until=current+0.65
                    for e in entities:
                        count+=1
                        event=f'#{count} {e.reason} [{e.start_t:.2f}s-{e.end_t:.2f}s]'
                        events.append(event)
                        print('[SEGMENT]',event, f'duration={e.end_t-e.start_t:.2f}s')
                out=stream.last_out
                p=float(torch.sigmoid(out['presence'][0]).item());q=float(torch.sigmoid(out['end'][0]).item())
                hist_p.append(p);hist_q.append(q)
                process_elapsed=max(time.perf_counter()-loop_start,1e-6)
                fps=0.85*fps+0.15*(1/process_elapsed) if fps else 1/process_elapsed
                display=mirrored.copy()
                draw_landmarks(display,parts)
                stats={'active':stream.tracker.active,'hands':int('left' in parts)+int('right' in parts),
                       'valid':int(valid.sum()),'ok':bool(out['hand_ok'][0].item()>0.5),
                       'speed':float(out['speed'][0].item()),'count':count,'fps':fps,
                       'just_cut':current<flash_until}
                viz=overlay(display,stats,hist_p,hist_q,events,cfg,show_panel)
                # View mirror toggle only changes output, not inference-coordinate meaning.
                if not mirror: viz[:display.shape[0],:display.shape[1]]=cv2.flip(viz[:display.shape[0],:display.shape[1]],1)
                cv2.imshow('YAP Phase A | Camera Segmentation Debugger',viz)
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),27):break
                if key==ord('d'):show_panel=not show_panel
                if key==ord('m'):mirror=not mirror
                if key==ord('s'):
                    filename=args.save_dir/f'yap_{int(time.time())}.png'
                    cv2.imwrite(str(filename),viz)
                    print('[SCREENSHOT]',filename)
                if key==ord('r'):
                    for e in stream.flush():
                        print('[FLUSH]',e.reason,e.start_t,e.end_t)
                    stream=model.stream(device=device)
                    hist_p.clear();hist_q.clear();events.clear();count=0;flash_until=0
                    print('[RESET] Stream state cleared')
        finally:
            cap.release();cv2.destroyAllWindows()
    print('[SUMMARY] completed segments:',count)


if __name__=='__main__':
    main()
