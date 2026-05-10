import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from extractKeypoints import extract_raw, interpolate, normalize, extractFeature

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 스크립트 위치를 기준으로 하위 폴더인 'dataset' 폴더의 경로를 잡습니다.
BASE_DIR = os.path.join(CURRENT_DIR, "dataset")

DIR_VIDEOS = os.path.join(BASE_DIR, "raw_videos")
DIR_RAW_JSON = os.path.join(BASE_DIR, "extracted_raw")
DIR_INTERP_JSON = os.path.join(BASE_DIR, "interpolated")
DIR_NORM_JSON = os.path.join(BASE_DIR, "normalized")
DIR_DOMINANT_JSON = os.path.join(BASE_DIR, "dominant_hand")

# 최상위 기본 폴더 4개 생성
for folder in [DIR_VIDEOS, DIR_RAW_JSON, DIR_INTERP_JSON, DIR_NORM_JSON, DIR_DOMINANT_JSON]:
    os.makedirs(folder, exist_ok=True)

def save_to_json(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # print(f"  -> 저장 완료: {file_path}") # 필요시 주석 해제하여 로그 확인

def process_video_to_final_json(video_path, current_idx, total_count):
    # 원본 비디오의 최상위 폴더(DIR_VIDEOS)를 기준으로 상대 경로 추출
    rel_path = os.path.relpath(video_path, DIR_VIDEOS)
    rel_dir = os.path.dirname(rel_path) # 하위 폴더 경로
    file_name = os.path.basename(video_path) # 파일 이름
    base_name, _ = os.path.splitext(file_name) # 확장자 제거

    # 각 단계별 최상위 폴더에 하위 폴더 경로 합치기
    raw_target_dir = os.path.join(DIR_RAW_JSON, rel_dir)
    interp_target_dir = os.path.join(DIR_INTERP_JSON, rel_dir)
    norm_target_dir = os.path.join(DIR_NORM_JSON, rel_dir)
    dominant_target_dir = os.path.join(DIR_DOMINANT_JSON, rel_dir)

    # 파일 저장 직전에, 목적지 폴더들이 실제로 존재하는지 확인하고 없으면 생성
    os.makedirs(raw_target_dir, exist_ok=True)
    os.makedirs(interp_target_dir, exist_ok=True)
    os.makedirs(norm_target_dir, exist_ok=True)
    os.makedirs(dominant_target_dir, exist_ok=True)

    # 최종 파일 저장 경로
    raw_json_path = os.path.join(raw_target_dir, f"{base_name}_keypoints_raw.json")
    interp_json_path = os.path.join(interp_target_dir, f"{base_name}_keypoints_interp.json")
    norm_json_path = os.path.join(norm_target_dir, f"{base_name}_keypoints_norm.json")
    dominant_json_path = os.path.join(dominant_target_dir, f"{base_name}_final.json")

    # # 진행 상황바 생성
    # with tqdm(total=3, desc=f"처리 중: {rel_dir}/{base_name}", leave=False) as pbar:
    #     pbar.set_description(f"[1/3] raw keypoint 추출 중...")
    #     raw_data = extract_raw.extract_raw_features(video_path)
    #     # 파일 저장할 필요 없다 생각되면 아래 줄 코드 주석 처리하기
    #     save_to_json(raw_data, raw_json_path)
        
    #     if not raw_data:
    #         print(f"[ERROR] 데이터 추출 실패 (영상 확인 필요: {rel_path})")
    #         return
        
    #     pbar.update(1)
        
    #     pbar.set_description(f"[2/3] 결측치 보간 중...")
    #     interpolated_data = interpolate.interpolate_landmarks(raw_data, limit_frames=10)
    #     # 파일 저장할 필요 없다 생각되면 아래 줄 코드 주석 처리하기
    #     save_to_json(interpolated_data, interp_json_path)
        
    #     pbar.update(1)

    #     pbar.set_description(f"[3/3] 정규화 중...")
    #     normalized_data = [normalize.normalize_frame(frame) for frame in interpolated_data]
    #     save_to_json(normalized_data, norm_json_path)

    #     pbar.update(1)

    print(f"\n▶️ [{current_idx}/{total_count}] 시작: {rel_dir}/{base_name}")
    
    # [Step 1] raw 데이터 추출
    if os.path.exists(raw_json_path):
        print(f"  -> [1/3] ⏩ 기존 Raw 데이터 존재 (불러오기 완료)")
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        print(f"  -> [1/3] ⚙️ 추출 중...")
        raw_data = extract_raw.extract_raw_features(video_path)
        save_to_json(raw_data, raw_json_path)
    
    if not raw_data:
        print(f"  ❌ [ERROR] 데이터 추출 실패 (영상 확인 필요): {rel_path}")
        return
        
    # [Step 2] 결측치 보간
    if os.path.exists(interp_json_path):
        print(f"  -> [2/3] ⏩ 기존 보간 데이터 존재 (불러오기 완료)")
        with open(interp_json_path, 'r', encoding='utf-8') as f:
            interpolated_data = json.load(f)
    else:
        print(f"  -> [2/3] ⚙️ 결측치 보간 중...")
        interpolated_data = interpolate.interpolate_landmarks(raw_data, limit_frames=10)
        save_to_json(interpolated_data, interp_json_path)

    
    # [Step 3] 정규화
    if os.path.exists(norm_json_path):
        print(f"  -> [3/3] ⏩ 기존 정규화 데이터 존재 (건너뜀)")
        with open(norm_json_path, 'r', encoding='utf-8') as f:
            normalized_data = json.load(f)
    else:
        print(f"  -> [3/3] ⚙️ 정규화 중...")
        normalized_data = [normalize.normalize_frame(frame) for frame in interpolated_data]
        save_to_json(normalized_data, norm_json_path)

    # [Step 4] 우세손 판별
    if os.path.exists(dominant_json_path):
        print(f"  -> [4/4] ⏩ 기존 우세손 데이터 존재 (스킵)")
    else:
        print(f"  -> [4/4] ⚙️ 우세손 판별 및 최종 패키징 중...")
        
        # 앞서 만든 함수 호출 (리스트 형태의 normalized_data를 그대로 전달)
        dominant_data = extractFeature.determine_dominant_hand(normalized_data)
        
        final_package = {
            "metadata": {
                "dominant_hand": dominant_data,
                "total_frames": len(normalized_data),
                "video_fileName": base_name
            },
            "frames": normalized_data
        }
        save_to_json(final_package, dominant_json_path)

    print(f"  ✅ 완료!")


# 2. 파이프라인 실행 함수
# def run_pipeline():
#     print(f"🔍 '{DIR_VIDEOS}' 폴더 내부의 모든 하위 폴더를 스캔")

#     video_files = []
#     # os.walk를 사용하면 아무리 폴더가 깊게 중첩되어 있어도 모든 파일을 긁어옵니다.
#     for root, dirs, files in os.walk(DIR_VIDEOS):
#         for file in files:
#             # 대소문자 구분 없이 mp4 확장자 찾기 (예: .MP4, .mp4)
#             if file.lower().endswith(".mp4"):
#                 video_files.append(os.path.join(root, file))
    
#     if len(video_files) == 0:
#         print(f"⚠️ 변환할 mp4 파일이 없습니다. 폴더 구조를 확인해 주세요.")
#         return
        
#     print(f"🚀 총 {len(video_files)}개의 영상을 찾았습니다. 전체 파이프라인을 가동합니다.\n")

#     for video_path in tqdm(video_files, desc="전체 파이프라인 진행률", position=0):
#         process_video_to_final_json(video_path)

#     print("\n🎉 전체 데이터셋 구축 및 폴더 미러링이 완벽하게 완료되었습니다!")


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