# convert_sensor_pipeline.py
import cv2
import json
import os
import numpy as np
import copy
from convertKeypoints.pseudo_sensor import PseudoSensorConverter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 스크립트 위치를 기준으로 하위 폴더인 'dataset' 폴더의 경로를 잡습니다.
BASE_DIR = os.path.join(CURRENT_DIR, "dataset")

DIR_VIDEOS = os.path.join(CURRENT_DIR, "raw_videos")

# 최상위 폴더 경로 설정
TRANSLATION_BASE_DIR = os.path.join(BASE_DIR, "translation_dataset")
TRANSLATION_SENSOR_DIR = os.path.join(TRANSLATION_BASE_DIR, "converted_sensor")
TRANSLATION_FINAL_DIR = os.path.join(TRANSLATION_BASE_DIR, "final_dataset")

KEYPOINT_GROUPS = [
    'face_keypoints', 'lip_keypoints', 'eyebrow_keypoints', 
    'eye_keypoints', 'pose_keypoints', 'left_hand_keypoints', 'right_hand_keypoints'
]

# 최상위 기본 폴더들 자동 생성
for folder in [TRANSLATION_BASE_DIR, TRANSLATION_SENSOR_DIR, TRANSLATION_FINAL_DIR]:
    os.makedirs(folder, exist_ok=True)


def save_to_json(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def process_keypoints_to_seonsor_json(video_path, current_idx, total_count):
    # 원본 비디오의 최상위 폴더(DIR_VIDEOS)를 기준으로 상대 경로 추출
    rel_path = os.path.relpath(video_path, DIR_VIDEOS)
    rel_dir = os.path.dirname(rel_path) # 하위 폴더 경로
    file_name = os.path.basename(video_path) # 파일 이름
    base_name, _ = os.path.splitext(file_name) # 확장자 제거

    # 각 단계별 최상위 폴더에 하위 폴더 경로 합치기
    trans_sensor_target_dir = os.path.join(TRANSLATION_SENSOR_DIR, rel_dir)
    trans_final_target_dir = os.path.join(TRANSLATION_FINAL_DIR, rel_dir)

    # 파일 저장 직전에, 목적지 폴더들이 실제로 존재하는지 확인하고 없으면 생성
    for d in [trans_sensor_target_dir, trans_final_target_dir]:
        os.makedirs(d, exist_ok=True)

    # 최종 파일 저장 경로
    trans_sensor_json_path = os.path.join(trans_sensor_target_dir, f"{base_name}_converted_sensor.json")
    trans_final_json_path = os.path.join(trans_final_target_dir, f"{base_name}_keypoints_final.json")

    print(f"\n▶️ [{current_idx}/{total_count}] 시작: {rel_dir}/{base_name}")
    
    if not os.path.exists(trans_final_json_path):
        print(f"  ⚠️ [경고] 원본 JSON 파일이 없습니다. 건너뜁니다: {trans_final_json_path}")
        return
    
    if os.path.exists(trans_sensor_json_path):
        print(f"  -> ⏩ 이미 변환된 센서 데이터가 존재합니다. (건너뜀)")
        return

    converted_data = PseudoSensorConverter.convert_to_sensor_data(input_json_path=trans_final_json_path)
    save_to_json(converted_data, trans_sensor_json_path)
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
            process_keypoints_to_seonsor_json(video_path, i + 1, total_files)
        except Exception as e:
            print(f"  ❌ [치명적 에러] {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 전체 데이터셋 구축이 완료되었습니다!")

if __name__ == "__main__":
    run_pipeline()