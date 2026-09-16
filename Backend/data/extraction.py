import numpy as np

def extract_frame_features(hands_result, pose_result, face_result):
    """Canonical 163-dim feature extraction, shared by ALL stages (word image / sentence frame / video frame).

    Layout (must match MultiStreamEmbedding slicing exactly):
      [0:63]    lh (21 landmarks x 3)
      [63:126]  rh (21 landmarks x 3)
      [126:138] arms (4 landmarks x 3)  -> shoulders(11,12) + elbows(13,14)
      [138:159] face (7 landmarks x 3)  -> eyes/nose/cheeks/mouth corners
      [159:163] flags [lh_flag, rh_flag, arms_flag, face_flag]
    """
    # --- Hands ---
    lh = np.zeros((21, 3)); rh = np.zeros((21, 3))
    lh_flag = 0.0; rh_flag = 0.0
    if hands_result and hands_result.hand_landmarks:
        for idx, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hands_result.hand_landmarks[idx]])
            if label == 'Left':
                lh = pts; lh_flag = 1.0
            else:
                rh = pts; rh_flag = 1.0

    # --- Arms (shoulders + elbows) ---
    arms = np.zeros((4, 3))
    arms_flag = 0.0
    if pose_result and pose_result.pose_landmarks:
        p = pose_result.pose_landmarks[0]
        arms = np.array([
            [p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
            [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z],
        ])
        arms_flag = 1.0

    # --- Face (7 pts: eyes, nose tip, cheeks, mouth corners) ---
    face_pts = np.zeros((7, 3))
    face_flag = 0.0
    if face_result and face_result.face_landmarks:
        f = face_result.face_landmarks[0]
        face_pts = np.array([
            [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
            [f[4].x, f[4].y, f[4].z],
            [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
            [f[61].x, f[61].y, f[61].z],    [f[291].x, f[291].y, f[291].z],
        ])
        face_flag = 1.0

    # --- Normalization: anchor to nose, scale by shoulder width (PER-FRAME, not sequence-mean) ---
    anchor = face_pts[2] if face_flag else np.array([0.0, 0.0, 0.0])
    if lh_flag: lh = lh - anchor
    if rh_flag: rh = rh - anchor
    if arms_flag: arms = arms - anchor
    if face_flag: face_pts = face_pts - anchor

    scale = np.linalg.norm(arms[0] - arms[1]) if arms_flag else 0.0
    if scale > 0.05:
        lh = lh / scale; rh = rh / scale; arms = arms / scale; face_pts = face_pts / scale

    frame_data = np.concatenate([
        lh.flatten(), rh.flatten(), arms.flatten(), face_pts.flatten(),
        [lh_flag, rh_flag, arms_flag, face_flag]
    ])
    return frame_data.astype(np.float32)  # (163,)


def extract_frame_features_video_mode(hands_lh, hands_rh, hands_lh_flag, hands_rh_flag,
                                       pose_landmarks, face_landmarks,
                                       last_lh, last_rh, last_arms, last_face):
    """Variant for VIDEO streams: carries forward last-known landmarks on tracking dropout
    instead of zeroing (prevents 'teleporting' hands / spurious zero-frames that corrupt
    per-frame normalization). Returns (frame_features, new_last_lh, new_last_rh, new_last_arms, new_last_face).
    """
    lh = hands_lh if hands_lh_flag else last_lh
    rh = hands_rh if hands_rh_flag else last_rh
    lh_flag = 1.0 if hands_lh_flag else (1.0 if np.any(last_lh != 0) else 0.0)
    rh_flag = 1.0 if hands_rh_flag else (1.0 if np.any(last_rh != 0) else 0.0)

    if pose_landmarks:
        p = pose_landmarks[0] if isinstance(pose_landmarks[0], list) else pose_landmarks
        arms = np.array([[p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
                          [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z]])
        arms_flag = 1.0
    else:
        arms = last_arms
        arms_flag = 1.0 if np.any(last_arms != 0) else 0.0

    if face_landmarks:
        f = face_landmarks[0] if isinstance(face_landmarks[0], list) else face_landmarks
        face_pts = np.array([
            [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
            [f[4].x, f[4].y, f[4].z],
            [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
            [f[61].x, f[61].y, f[61].z],    [f[291].x, f[291].y, f[291].z],
        ])
        face_flag = 1.0
    else:
        face_pts = last_face
        face_flag = 1.0 if np.any(last_face != 0) else 0.0

    anchor = face_pts[2] if face_flag else np.array([0.0, 0.0, 0.0])
    lh2 = lh - anchor if lh_flag else lh
    rh2 = rh - anchor if rh_flag else rh
    arms2 = arms - anchor if arms_flag else arms
    face2 = face_pts - anchor if face_flag else face_pts

    scale = np.linalg.norm(arms2[0] - arms2[1]) if arms_flag else 0.0
    if scale > 0.05:
        lh2 = lh2 / scale; rh2 = rh2 / scale; arms2 = arms2 / scale; face2 = face2 / scale

    frame_data = np.concatenate([
        lh2.flatten(), rh2.flatten(), arms2.flatten(), face2.flatten(),
        [lh_flag, rh_flag, arms_flag, face_flag]
    ]).astype(np.float32)

    return frame_data, lh, rh, arms, face_pts
