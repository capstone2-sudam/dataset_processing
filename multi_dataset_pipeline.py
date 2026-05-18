import cv2
import json
import os
import numpy as np
import copy
from convert_rawKeypoints import convert_to_normalized_coordinates
from extractKeypoints.interpolate import interpolate_landmarks
from extractKeypoints.normalize import normalize_frame_translation, smooth_avatar_frames

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 스크립트 위치를 기준으로 하위 폴더인 'dataset' 폴더의 경로를 잡습니다.
BASE_DIR = os.path.join(CURRENT_DIR, "dataset")

# 최상위 폴더 경로 설정
TRANSLATION_BASE_DIR = os.path.join(BASE_DIR, "translation_dataset")
UNITY_BASE_DIR = os.path.join(BASE_DIR, "avatar_dataset")

DIR_VIDEOS = os.path.join(CURRENT_DIR, "raw_videos")
TRANSLATION_RAW_DIR = os.path.join(TRANSLATION_BASE_DIR, "extracted_raw")
TRANSLATION_INTERP_DIR = os.path.join(TRANSLATION_BASE_DIR, "interpolated")
TRANSLATION_FINAL_DIR = os.path.join(TRANSLATION_BASE_DIR, "final_dataset")

UNITY_RAW_DIR = os.path.join(UNITY_BASE_DIR, "extracted_raw")
UNITY_INTERP_DIR = os.path.join(UNITY_BASE_DIR, "interpolated")
UNITY_FINAL_DIR = os.path.join(UNITY_BASE_DIR, "final_dataset")

KEYPOINT_GROUPS = [
    'face_keypoints', 'lip_keypoints', 'eyebrow_keypoints', 
    'eye_keypoints', 'pose_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'
]

# 최상위 기본 폴더들 자동 생성
for folder in [DIR_VIDEOS, TRANSLATION_BASE_DIR, UNITY_BASE_DIR, TRANSLATION_RAW_DIR, TRANSLATION_INTERP_DIR, TRANSLATION_FINAL_DIR, 
               UNITY_RAW_DIR, UNITY_INTERP_DIR, UNITY_FINAL_DIR]:
    os.makedirs(folder, exist_ok=True)


def save_to_json(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def process_video_to_final_json(video_path, current_idx, total_count):
    # 원본 비디오의 최상위 폴더(DIR_VIDEOS)를 기준으로 상대 경로 추출
    rel_path = os.path.relpath(video_path, DIR_VIDEOS)
    rel_dir = os.path.dirname(rel_path) # 하위 폴더 경로
    file_name = os.path.basename(video_path) # 파일 이름
    base_name, _ = os.path.splitext(file_name) # 확장자 제거

    # 각 단계별 최상위 폴더에 하위 폴더 경로 합치기
    trans_raw_target_dir = os.path.join(TRANSLATION_RAW_DIR, rel_dir)
    trans_interp_target_dir = os.path.join(TRANSLATION_INTERP_DIR, rel_dir)
    trans_final_target_dir = os.path.join(TRANSLATION_FINAL_DIR, rel_dir)

    unity_raw_target_dir = os.path.join(UNITY_RAW_DIR, rel_dir)
    unity_interp_target_dir = os.path.join(UNITY_INTERP_DIR, rel_dir)
    unity_final_target_dir = os.path.join(UNITY_FINAL_DIR, rel_dir)


    # 파일 저장 직전에, 목적지 폴더들이 실제로 존재하는지 확인하고 없으면 생성
    for d in [trans_raw_target_dir, trans_interp_target_dir, trans_final_target_dir,
              unity_raw_target_dir, unity_interp_target_dir, unity_final_target_dir]:
        os.makedirs(d, exist_ok=True)

    # 최종 파일 저장 경로
    trans_raw_json_path = os.path.join(trans_raw_target_dir, f"{base_name}_keypoints_raw.json")
    trans_interp_json_path = os.path.join(trans_interp_target_dir, f"{base_name}_keypoints_interp.json")
    trans_final_json_path = os.path.join(trans_final_target_dir, f"{base_name}_keypoints_final.json")

    unity_raw_json_path = os.path.join(unity_raw_target_dir, f"{base_name}_keypoints_raw.json")
    unity_interp_json_path = os.path.join(unity_interp_target_dir, f"{base_name}_keypoints_interp.json")
    unity_final_json_path = os.path.join(unity_final_target_dir, f"{base_name}_keypoints_final.json")

    print(f"\n▶️ [{current_idx}/{total_count}] 시작: {rel_dir}/{base_name}")
    
    # ===========================================================================

    # [Step 1] raw 데이터 추출
    if os.path.exists(trans_raw_json_path) and os.path.exists(unity_raw_json_path):
        print(f"  -> [1/3] ⏩ 기존 Raw 데이터 존재 (불러오기 완료)")
        
    else:
        print(f"  -> [1/3] 추출 및 복원 연산 실행 중...")
        convert_to_normalized_coordinates(video_path, current_idx, total_count)

    with open(trans_raw_json_path, 'r', encoding='utf-8') as f:
            trans_raw_data = json.load(f)
    with open(unity_raw_json_path, 'r', encoding='utf-8') as f:
        unity_raw_data = json.load(f)

    if not trans_raw_data or not unity_raw_data:
        print(f"  ❌ [ERROR] 데이터 추출 실패 (영상 확인 필요): {rel_path}")
        return
    
    # =========================================================================== 

    # [Step 2-1] 번역용 데이터 보간 (구조: 순수 프레임 리스트)
    if os.path.exists(trans_interp_json_path):
        print(f"  --> [2/3] ⏩ 기존 번역용 보간 데이터 존재")
        with open(trans_interp_json_path, 'r', encoding='utf-8') as f:
            trans_interp_data = json.load(f)
    else:
        print(f"  --> [2/3] 번역용 결측치 보간 중...")
        # 번역용은 데이터 자체가 리스트이므로 통째로 대입
        trans_interp_data = interpolate_landmarks(trans_raw_data, limit_frames=5)
        save_to_json(trans_interp_data, trans_interp_json_path)

    # [Step 2-2] 아바타용 데이터 보간 (구조: 메타데이터 패키지 딕셔너리)
    if os.path.exists(unity_interp_json_path):
        print(f"  --> [2/3] ⏩ 기존 아바타용 보간 데이터 존재")
        with open(unity_interp_json_path, 'r', encoding='utf-8') as f:
            unity_interp_package = json.load(f)
    else:
        print(f"  --> [2/3] 아바타용 결측치 보간 중...")
        # ✨ 비결: 딕셔너리 껍데기에서 "frames" 알맹이 리스트만 쏙 추출해서 전달!
        raw_unity_frames = unity_raw_data["frames"]
        unity_interp_frames = interpolate_landmarks(raw_unity_frames, limit_frames=5)
        
        # 원본 메타데이터 구조 복사 후 보간된 데이터로 알맹이만 교체 (Wrapping)
        unity_interp_package = copy.deepcopy(unity_raw_data)
        unity_interp_package["frames"] = unity_interp_frames
        save_to_json(unity_interp_package, unity_interp_json_path)

    # =============================================================================================

    # [Step 3.1] 번역용 데이터 정규화
    if os.path.exists(trans_final_json_path):
        print(f"  --> [3/3] ⏩ 기존 번역용 정규화 데이터 존재")

    else:
        print(f"  --> [3/3] 번역용 데이터 정규화 수행 중...")
        # 번역용은 데이터 자체가 리스트이므로 통째로 대입
        trans_normalized_data = [normalize_frame_translation(frame) for frame in trans_interp_data]
        
        korean_text = ""
        gt_json_path = os.path.join(os.path.dirname(video_path), f"{base_name}.json")
        
        if os.path.exists(gt_json_path):
            try:
                with open(gt_json_path, 'r', encoding='utf-8') as f:
                    gt_data = json.load(f)
                    # 딕셔너리의 안전한 접근(get)을 사용하여 에러 방지
                    korean_text = gt_data.get("krlgg_sntenc", {}).get("koreanText", "")
            except Exception as e:
                print(f"  ⚠️ [경고] 정답 JSON 파일을 읽는 중 오류 발생 ({base_name}.json): {e}")
        else:
            print(f"  ⚠️ [경고] 짝이 맞는 정답 JSON 파일이 없습니다: {base_name}.json")
        
        translation_package = {
         "metadata": {
            "total_frames": len(trans_normalized_data),
            "video_fileName": base_name,
            "fps": 30,
            "koreanText": korean_text
        },
        "frames": trans_normalized_data
    }
        save_to_json(translation_package, trans_final_json_path)

    # [Step 3-2] 아바타용 데이터 스무딩
    if os.path.exists(unity_final_json_path):
        print(f"  --> [3/3] ⏩ 기존 아바타용 평활화 데이터 존재")

    else:
        print(f"  --> [3/3] 아바타용 데이터 평활화(Smoothing) 수행 중...")
        # ✨ 비결: 딕셔너리 껍데기에서 "frames" 알맹이 리스트만 쏙 추출해서 전달!
        unity_interp_frames = unity_interp_package["frames"]
        unity_processed_frames = smooth_avatar_frames(unity_interp_frames, window_size=3)
        
        # 원본 메타데이터 구조 복사 후 보간된 데이터로 알맹이만 교체 (Wrapping)
        unity_processed_package = copy.deepcopy(unity_raw_data)
        unity_processed_package["frames"] = unity_processed_frames
        save_to_json(unity_processed_package, unity_final_json_path)

    print(f"  ✅ 완료!")


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
            process_video_to_final_json(video_path, i + 1, total_files)
        except Exception as e:
            print(f"  ❌ [치명적 에러] {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 전체 데이터셋 구축이 완료되었습니다!")

if __name__ == "__main__":
    run_pipeline()