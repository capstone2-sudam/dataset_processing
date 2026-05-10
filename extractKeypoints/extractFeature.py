# 우세손 추출 (의미적 특징 추출)
# 손목 좌표의 공간 표준편차 이용 (데이터가 평균으로부터 얼마나 넓게 퍼져 있는가)
# 전체 dataset_pipeline을 통해 물리적인 특징 추출 후에 추가 수행

import numpy as np

'''
# 기존 리스트 구조 앞에 메타데이터 정보를 포함하는 딕셔너리로 감싸는 방식
output_data = {
    "metadata": {
        "dominant_hand": "Right",
        "total_frames": 109,
        "video_name": "NIA_SL_FS0001"
    },
    "frames": normalized_data  # 기존 프레임 리스트
}
'''

def determine_dominant_hand(frames, noise_threshold=0.03, ratio_limit=1.5):
    left_wrist_path = []
    right_wrist_path = []
    
    for f in frames:
        l_pts = np.array(f.get('left_hand_keypoints', []))
        if len(l_pts) > 0 and not np.all(l_pts == 0):
            left_wrist_path.append(l_pts[0])
            
        r_pts = np.array(f.get('right_hand_keypoints', []))
        if len(r_pts) > 0 and not np.all(r_pts == 0):
            right_wrist_path.append(r_pts[0])

    # 공간 표준편차 구하는 함수
    def get_activity(path):
        if len(path) < 5: 
            return 0.0
        path = np.array(path)
        return float(np.sum(np.std(path, axis=0)))

    left_activity = get_activity(left_wrist_path)
    right_activity = get_activity(right_wrist_path)

    # 노이즈 임계점보다 값이 작으면 0.0으로
    if left_activity < noise_threshold: left_activity = 0.0
    if right_activity < noise_threshold: right_activity = 0.0

    if left_activity == 0 and right_activity == 0: 
        return "None"
    elif left_activity == 0: 
        return "Right"
    elif right_activity == 0: 
        return "Left"
        
    ratio = right_activity / left_activity
    if ratio > ratio_limit: 
        return "Right"
    elif ratio < (1.0 / ratio_limit): 
        return "Left"
    else: 
        return "Both"