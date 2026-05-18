import cv2
import json
import os
import numpy as np
import copy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 최상위 폴더 경로 설정
TRANSLATION_BASE_DIR = os.path.join(CURRENT_DIR, "translation_dataset")
UNITY_BASE_DIR = os.path.join(CURRENT_DIR, "avatar_dataset")

DIR_VIDEOS = os.path.join(CURRENT_DIR, "raw_videos")
DIR_ORIGINAL_RAW_JSON = os.path.join(CURRENT_DIR, "extracted_original_raw")
DIR_TRANSLATION_RAW_JSON = os.path.join(TRANSLATION_BASE_DIR, "extracted_raw")
DIR_UNITY_RAW_JSON = os.path.join(UNITY_BASE_DIR, "extracted_raw")

'''
파일 구조
current_dir / translation_dataset / extracted_raw
current_dir / avator_dataset / extracted_raw
current_dir / raw_videos (translation과 avator dataset은 같은 영상 데이터를 사용)
'''

KEYPOINT_GROUPS = [
    'face_keypoints', 'lip_keypoints', 'eyebrow_keypoints', 
    'eye_keypoints', 'pose_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'
]

# 최상위 기본 폴더 4개 생성
for folder in [DIR_VIDEOS, DIR_ORIGINAL_RAW_JSON, DIR_TRANSLATION_RAW_JSON, DIR_UNITY_RAW_JSON]:
    os.makedirs(folder, exist_ok=True)

def read_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    return width, height

def save_to_json(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # print(f"  -> 저장 완료: {file_path}") # 필요시 주석 해제하여 로그 확인


def convert_to_normalized_coordinates(video_path, current_idx, total_count):
    # 원본 비디오의 최상위 폴더(DIR_VIDEOS)를 기준으로 상대 경로 추출
    rel_path = os.path.relpath(video_path, DIR_VIDEOS)
    rel_dir = os.path.dirname(rel_path) # 하위 폴더 경로 (예: 01_NIKL_...)
    file_name = os.path.basename(video_path) # 파일 이름 (예: VXPAKOKS230000010.mp4)
    base_name, _ = os.path.splitext(file_name) # 확장자 제거

    # 원본 픽셀 JSON 경로 계산
    src_json_path = os.path.join(DIR_ORIGINAL_RAW_JSON, rel_dir, f"{base_name}_keypoints_raw.json")

    # 원본 JSON 파일이 없으면 매칭 실패로 간주하고 스킵
    if not os.path.exists(src_json_path):
        print(f"  ⚠️ [스킵] 원본 JSON 없음: {rel_dir}/{base_name}_keypoints_raw.json")
        return    
    
    # 각 단계별 최상위 폴더에 하위 폴더 경로 합치기
    raw_translation_dir = os.path.join(DIR_TRANSLATION_RAW_JSON, rel_dir)
    raw_unity_dir = os.path.join(DIR_UNITY_RAW_JSON, rel_dir)
    
    # 파일 저장 직전에, 목적지 폴더들이 실제로 존재하는지 확인하고 없으면 생성
    os.makedirs(raw_translation_dir, exist_ok=True)
    os.makedirs(raw_unity_dir, exist_ok=True)

    # 최종 파일 저장 경로
    raw_translation_json_path = os.path.join(raw_translation_dir, f"{base_name}_keypoints_raw.json")
    raw_unity_json_path = os.path.join(raw_unity_dir, f"{base_name}_keypoints_raw.json")

    print(f"\n▶️ [{current_idx}/{total_count}] 시작: {rel_dir}/{base_name}")

    # 1. 영상에서 해상도(Metadata) 가져오기
    width, height = read_metadata(video_path)
    if width == 0 or height == 0:
        print(f"  ❌ [ERROR] 영상 해상도를 읽지 못했습니다: {file_name}")
        return

    # 2. 기존 픽셀 기반 원본 JSON 로드
    with open(src_json_path, 'r', encoding='utf-8') as f:
        original_frames = json.load(f)

    # 3. 데이터 복원 연산 수행 (Deep Copy로 각각 독립된 데이터 생성)
    translation_frames = copy.deepcopy(original_frames)
    unity_frames = copy.deepcopy(original_frames)

    for t_frame, u_frame in zip(translation_frames, unity_frames):
        for group in KEYPOINT_GROUPS:
            # translation 데이터 처리: z축만 width로 나누기
            t_pts = t_frame.get(group, [])
            if t_pts:
                t_arr = np.array(t_pts, dtype=float)
                if not np.all(t_arr == 0):
                    t_arr[:, 2] /= width # z축 복원
                    t_frame[group] = np.round(t_arr, 6).tolist()
            
            # unity avatar 데이터 처리: x, y, z 전축 복원
            u_pts = u_frame.get(group, [])
            if u_pts:
                u_arr = np.array(u_pts, dtype=float)
                if not np.all(u_arr == 0):
                    u_arr[:, 0] /= width
                    u_arr[:, 1] /= height
                    u_arr[:, 2] /= width
                    u_frame[group] = np.round(u_arr, 6).tolist()

    translation_package = {
         "metadata": {
            "total_frames": len(translation_frames),
            "video_fileName": base_name,
            "video_width": width,
            "video_height": height,
            "fps": 30
        },
        "frames": translation_frames
    }

    unity_package = {
        "metadata": {
            "total_frames": len(unity_frames),
            "video_fileName": base_name,
            "video_width": width,
            "video_height": height,
            "fps": 30
        },
        "frames": unity_frames
    }

    save_to_json(translation_package, raw_translation_json_path)
    save_to_json(unity_package, raw_unity_json_path)
    print(f"  ✅ 완료 (Translation & Avatar 데이터 분리 저장 성공)")


def run_pipeline():
    print(f"🔍 '{DIR_VIDEOS}' 폴더 스캔 중...")

    video_files = []
    for root, dirs, files in os.walk(DIR_VIDEOS):
        for file in files:
            if file.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, file))
    
    total_files = len(video_files)
    if total_files == 0:
        print(f"⚠️ 변환할 mp4 파일이 없습니다.")
        return
        
    print(f"🚀 총 {total_files}개의 영상을 찾았습니다. 파이프라인을 가동합니다.\n")

    for i, video_path in enumerate(video_files):
        try:
            convert_to_normalized_coordinates(video_path, i + 1, total_files)
        except Exception as e:
            print(f"  ❌ [치명적 에러] {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 전체 데이터셋 구축이 완료되었습니다!")

if __name__ == "__main__":
    run_pipeline()