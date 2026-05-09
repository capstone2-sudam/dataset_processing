import numpy as np
import pandas as pd

KEYPOINT_SIZES = {
    'face_keypoints': 9,
    'lip_keypoints': 40,
    'eyebrow_keypoints': 10,
    'eye_keypoints': 32,
    'pose_keypoints': 8,
    'left_hand_keypoints': 21,
    'right_hand_keypoints': 21
}

# 추출된 프레임 데이터의 결측치([0.0, 0.0, 0.0])를 선형 보간
def interpolate_landmarks(extracted_data, limit_frames=10):
    if not extracted_data:
        return []
    
    # JSON(3D 데이터)을 Pandas DataFrame으로 평탄화 (Flatten)
    flattened_data = []
    for frame in extracted_data:
        row = {'frame_number': frame['frame_number'], 'timestamp': frame['timestamp']}
        
        for key in KEYPOINT_SIZES.keys():
            points = np.array(frame.get(key, []))
            
            if len(points) == 0:
                # 데이터가 아예 없는 경우 0 배열로 초기화
                points = np.zeros((KEYPOINT_SIZES[key], 3))

            # [x, y, z]가 모두 0인 경우를 찾아 NaN으로 치환
            mask = (points == 0).all(axis=1)
            points = points.astype(float)
            points[mask] = np.nan
            
            # x, y, z를 각각의 독립된 열(Column)로 펼침
            for i, (x, y, z) in enumerate(points):
                row[f'{key}_{i}_x'] = x
                row[f'{key}_{i}_y'] = y
                row[f'{key}_{i}_z'] = z
                
        flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    # 선형 보간법 적용
    cols = [c for c in df.columns if c not in ['frame_number', 'timestamp']]
    df[cols] = df[cols].interpolate(method='linear', limit=limit_frames, limit_area='inside')
        
    # 몸통(Pose)과 얼굴(Face, Lip, Eye) 컬럼 분류
    body_face_cols = [c for c in cols if 'pose' in c or 'face' in c or 'lip' in c or 'eye' in c]
    # 손(Hand) 컬럼 분류
    hand_cols = [c for c in cols if 'hand' in c]

    # 몸통과 얼굴은 영상 끝단에서도 존재해야 하므로 과거/미래 값으로 채움
    df[body_face_cols] = df[body_face_cols].bfill().ffill()

    # 손은 짧은 깜빡임(10프레임)은 위에서 보간되었으므로, 
    # 양 끝단의 긴 차렷 자세(NaN)는 억지로 채우지 않고 0.0으로 둠
    df[hand_cols] = df[hand_cols].fillna(0.0)

    interpolated_list = []
    for _, row in df.iterrows():
        frame_dict = {
            'frame_number': int(row['frame_number']),
            'timestamp': int(row['timestamp'])
        }
        for key, size in KEYPOINT_SIZES.items():
            pts = []
            for i in range(size):
                x, y, z = row[f'{key}_{i}_x'], row[f'{key}_{i}_y'], row[f'{key}_{i}_z']
                # 보간 후에도 남은 NaN(데이터가 아예 없는 영상 등)은 0.0으로 안전 처리
                pts.append([round(x, 6) if not np.isnan(x) else 0.0,
                            round(y, 6) if not np.isnan(y) else 0.0,
                            round(z, 6) if not np.isnan(z) else 0.0])
            frame_dict[key] = pts
        interpolated_list.append(frame_dict)

    return interpolated_list