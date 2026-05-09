import numpy as np

# 처리할 키포인트 그룹 리스트
KEYPOINT_GROUPS = [
    'face_keypoints', 'lip_keypoints', 'eyebrow_keypoints', 
    'eye_keypoints', 'pose_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'
]

def normalize_frame(frame):
    # 각 파트별 데이터를 Numpy 배열로 변환
    pose = np.array(frame.get('pose_keypoints', [])) # 손목 포함
    face_pose = np.array(frame.get('face_keypoints', [])) # Pose 모델에서 뽑은 코(0번) 포함
    left_hand = np.array(frame.get('left_hand_keypoints', []))
    right_hand = np.array(frame.get('right_hand_keypoints', []))
    lip = np.array(frame.get('lip_keypoints', []))
    eyebrow = np.array(frame.get('eyebrow_keypoints', []))
    eye = np.array(frame.get('eye_keypoints', []))

    # [Step 3-1: Z축 조립 (Anchor Merging)]
    # 왼손 (Pose 15번 손목 기준)
    # SELECTED_POSE_INDICES [11, 12, 13, 14, 15, 16, 23, 24]
    if len(pose) >= 6 and len(left_hand) > 0 and not np.all(left_hand == 0):
        # Pose 손목 좌표 - Hand 손목 좌표 = 3차원 이동 거리(Offset)
        left_offset = pose[4] - left_hand[0]
        left_hand += left_offset  # 손 전체 좌표를 진짜 손목 위치로 한 번에 이동

    # 2. 오른손 완벽 조립 (X, Y, Z 모두 Pose 5번 손목에 부착)
    if len(pose) >= 6 and len(right_hand) > 0 and not np.all(right_hand == 0):
        right_offset = pose[5] - right_hand[0]
        right_hand += right_offset


    # 얼굴 세부 부위 조립 (Pose 코 0번 기준)
    if len(face_pose) > 0:
        nose_z = face_pose[0, 2] # 코의 Z 좌표
        if len(lip) > 0: lip[:, 2] += nose_z
        if len(eyebrow) > 0: eyebrow[:, 2] += nose_z
        if len(eye) > 0: eye[:, 2] += nose_z

    # 임시 딕셔너리에 조립된 배열 묶기
    temp_frame = {
        'face_keypoints': face_pose,
        'lip_keypoints': lip,
        'eyebrow_keypoints': eyebrow,
        'eye_keypoints': eye,
        'pose_keypoints': pose,
        'left_hand_keypoints': left_hand,
        'right_hand_keypoints': right_hand
    }

    # [Step 3-2: 영점화(Centering) 및 스케일링(Scaling)]
    if len(pose) >= 2:
        left_shoulder = pose[0]
        right_shoulder = pose[1]

        # 기준 계산
        center = (left_shoulder + right_shoulder) / 2.0
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

        # 에러 방지 (어깨너비가 0이 되는 극단적인 경우)
        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0

        # 변환 (Transform): (좌표 - 중심) / 어깨너비
        for group in KEYPOINT_GROUPS:
            pts = temp_frame[group]
            if len(pts) > 0 and not np.all(pts == 0):
                norm_pts = (pts - center) / shoulder_dist
                temp_frame[group] = norm_pts
            else:
                # 0으로 채워진 데이터는 공식에 넣지 말고 그대로 0으로 둔다.
                temp_frame[group] = pts

    # [Step 3-3: 결과 포장 (JSON 저장용)]
    out_frame = {
        'frame_number': frame['frame_number'],
        'timestamp': frame['timestamp']
    }
    
    # 소수점 6자리로 반올림하여 용량 최적화 및 리스트 변환
    for group in KEYPOINT_GROUPS:
        pts = temp_frame[group]
        if len(pts) > 0:
            out_frame[group] = np.round(pts, 6).tolist()
        else:
            out_frame[group] = []

    return out_frame