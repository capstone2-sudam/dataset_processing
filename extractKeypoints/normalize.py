import numpy as np
import copy

# 처리할 키포인트 그룹 리스트
KEYPOINT_GROUPS = [
    'face_keypoints', 'lip_keypoints', 'eyebrow_keypoints', 
    'eye_keypoints', 'pose_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'
]

# 번역용 AI 모델 학습을 위한 강한 정규화 함수
# 어깨 중심 영점화 + 어깨 너비 기준 전체 스케일링 수행
def normalize_frame_translation(frame):

    out_frame = {'frame_number': frame['frame_number'], 'timestamp': frame['timestamp']}
    # 각 파트별 데이터를 Numpy 배열로 변환
    temp_data = {}
    for key in KEYPOINT_GROUPS:
        temp_data[key] = np.array(frame.get(key, []), dtype=np.float32)

    pose = temp_data['pose_keypoints']
    pose_structure_valid = (pose.ndim == 2 and pose.shape[1] == 3 and len(pose) >= 6)
    shoulders_valid = (
        pose_structure_valid and 
        not np.allclose(pose[0], 0.0) and 
        not np.allclose(pose[1], 0.0)
    )

    # [Step 3-1: 손목 및 얼굴 부위 앵커 조립 (Anchor Merging)]
    # 0: 좌측어깨, 1: 우측어깨, 4: 좌측손목, 5: 우측손목
    # SELECTED_POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
    if shoulders_valid:
        left_hand = temp_data['left_hand_keypoints']
        if left_hand.ndim == 2 and left_hand.shape[1] == 3 and len(left_hand) > 0 and not np.allclose(left_hand, 0.0):
            left_hand += (pose[4] - left_hand[0])

        right_hand = temp_data['right_hand_keypoints']
        if right_hand.ndim == 2 and right_hand.shape[1] == 3 and len(right_hand) > 0 and not np.allclose(right_hand, 0.0):
            right_hand += (pose[5] - right_hand[0])
    


        # [Step 3-2: 영점화(Centering) 및 스케일링(Scaling)]
        shoulder_center = (pose[0] + pose[1]) / 2.0
        shoulder_dist = np.linalg.norm(pose[0] - pose[1])
        shoulder_dist = max(shoulder_dist, 1e-6)

        # 변환 (Transform): (좌표 - 중심) / 어깨너비
        for key in KEYPOINT_GROUPS:
            pts = temp_data[key]

            if len(pts) == 0 or pts.ndim != 2 or pts.shape[1] != 3:
                out_frame[key] = []
                continue

            if np.allclose(pts, 0.0):
                out_frame[key] = pts.tolist()
                continue
            
            # 중심을 빼고 체형 크기로 나눔
            pts = (pts - shoulder_center) / shoulder_dist
            out_frame[key] = np.round(pts.astype(np.float32), 6).tolist()
    
    else:
        # 어깨가 추출되지 않은 경우 영점화 및 스케일링 없이 그대로 반환
        for key in KEYPOINT_GROUPS:
            pts = temp_data[key]
            if len(pts) > 0 and pts.ndim == 2 and pts.shape[1] == 3:
                out_frame[key] = np.round(pts, 6).tolist()
            else:
                out_frame[key] = []

    return out_frame

# 이미 0 ~ 1 정규화가 끝난 데이터를 받아 shape 방어와 포맷팅만 수행
# def prepare_unity_frame(frame):
#     out_frame = {
#         'frame_number': frame['frame_number'],
#         'timestamp': frame['timestamp']
#     }

#     for key in KEYPOINT_GROUPS:
#         pts = np.array(frame.get(key, []), dtype=np.float32)

#         # shape 안정성 체크
#         if not np.allclose(t_arr, 0.0, atol=1e-6) or pts.ndim != 2 or pts.shape[1] != 3:
#             out_frame[key] = []
#             continue

#         # 결측치 (0.0) 방어
#         if np.allclose(pts, 0.0):
#             out_frame[key] = pts.tolist()
#             continue

#         # 추가 수학 연산 없이 용량 최적화(소수점 6자리)만 수행
#         out_frame[key] = np.round(pts, 6).tolist()
    
#     return out_frame

# 프레인 전체 시퀀스를 받아 이동 평균 필터를 적용
# 결측치는 평균 연산에서 제외하여 안전하게 깍아내기
def smooth_avatar_frames(frames, window_size=3):
    if len(frames) < window_size:
        return frames
    
    smoothed = copy.deepcopy(frames)

    for key in KEYPOINT_GROUPS:
        # 시퀀스 전체를 3D 텐서로 변환 (프레임수, 관절수, 3)
        tensor = np.array([f.get(key, []) for f in frames], dtype=np.float32)

        if tensor.size == 0 or tensor.ndim != 3 or tensor.shape[2] !=3:
            continue

        smoothed_tensor = np.copy(tensor).astype(np.float32)

        for i in range(len(tensor)):
            start = max(0, i - window_size // 2)
            end = min(len(tensor), i + window_size // 2 + 1)

            window_data = tensor[start:end]

            # 결측치를 평균 계산에서 제외하기 위한 마스크 생성
            valid_mask = ~np.all(np.isclose(window_data, 0.0, atol=1e-6), axis=2)

            for k in range(tensor.shape[1]): # 각 관절(Keypoint)별로 수행
                valid_pts = window_data[:, k, :][valid_mask[:, k]]
                if len(valid_pts) > 0:
                    smoothed_tensor[i, k, :] = np.mean(valid_pts, axis=0)

        # 평활화된 데이터를 다시 리스트 형태로 업데이트
        for idx, f in enumerate(smoothed):
            if len(f.get(key, [])) > 0: # 원본에 데이터가 있었던 경우만 덮어쓰기
                f[key] = np.round(smoothed_tensor[idx], 6).tolist()   
    
    return smoothed