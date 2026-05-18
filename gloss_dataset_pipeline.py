# Animation_id 매핑 및 단어별 누적 idx 카운팅 처리
import os
import json
import pandas as pd
from extractKeypoints import sliceGloss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CURRENT_DIR, "dataset")

# raw_videos 안에 mp4와 json이 같이 있음. 같은 수어 영상에 대해서는 동일 제목 mp4와 json 파일을 가지고 있음
DIR_VIDEOS = os.path.join(BASE_DIR, "raw_videos")
DIR_FINAL_JSON = os.path.join(BASE_DIR, "final_dataset")
DIR_GLOSS_JSON = os.path.join(BASE_DIR, "gloss_dataset")
DIR_NMS_JSON = os.path.join(BASE_DIR, "nms_dataset")

# 엑셀 파일 경로
EXCEL_MAPPING_PATH = os.path.join(CURRENT_DIR, "animation_mapping.xlsx")

'''
파이썬 스크립트와 같은 폴더 위치에 gloss_mapping.xlsx 엑셀 파일 만들어서 넣어주기
A열 제목: Gloss_ID (예: 문제1)
B열 제목: Animation_ID (예: 203 or ANIM_PROBLEM_01)
-> 엑셀을 읽으려면 터미널에서 pandas, openpyxl 패키지 설치해주기
(pip install pandas openpyxl)
'''

for folder in [DIR_GLOSS_JSON, DIR_NMS_JSON]:
    os.makedirs(folder, exist_ok=True)

# excel 관련 함수 (Animation 처리)
# 엑셀을 읽어서 {glossID: AnimationID} 딕셔너리로 반환
def load_excel_mapping():
    if not os.path.exists(EXCEL_MAPPING_PATH):
        print(f"❌ [ERROR] 엑셀 파일이 없습니다! ({EXCEL_MAPPING_PATH})")
        return {}
    
    df = pd.read_excel(EXCEL_MAPPING_PATH)
    # glossID나 animationID 둘 중 하나라도 칸이 비어있는 불량 데이터가 있으면, 빼고 읽도록
    df = df.dropna(subset=['glossID', 'animationID']) 
    
    # animationID를 정수(int)로 확실하게 변환
    return dict(zip(df['glossID'], df['animationID'].astype(int)))

# 전체 json 파일을 읽고 업데이트된 딕셔너리를 다시 엑셀 파일로 저장
def save_excel_mapping(mapping_dict):
    df = pd.DataFrame({
        'glossID': list(mapping_dict.keys()),
        'animationID': list(mapping_dict.values())
    })
    df.to_excel(EXCEL_MAPPING_PATH, index=False)
    print(f"💾 엑셀 파일 업데이트 완료: {EXCEL_MAPPING_PATH}")

def save_to_json(data, file_path):
    if data:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # print(f"  -> 저장 완료: {file_path}") # 필요시 주석 해제하여 로그 확인

def process_and_save_gesture(anno_data, final_data, base_target_glosses, base_name, gloss_anim_map, current_max_id, new_gloss_count, gloss_counter):
    extracted_gesture = sliceGloss.slice_gestures(anno_data, final_data, base_target_glosses)

    for clip in extracted_gesture:
        final_gloss_id = clip["gloss_id"]

        if final_gloss_id not in gloss_anim_map:
            current_max_id += 1
            gloss_anim_map[final_gloss_id] = current_max_id
            new_gloss_count += 1
            print(f"  -> ✨ 새로운 방향 동사 발견: {final_gloss_id} (ID: {current_max_id})")
        
        gloss_counter[final_gloss_id] = gloss_counter.get(final_gloss_id, 0) + 1
        current_idx = gloss_counter[final_gloss_id]

        save_dir = os.path.join(DIR_GLOSS_JSON, final_gloss_id)
        os.makedirs(save_dir, exist_ok=True)

        # 파일명 규칙: glossID_순번_원본명.json (우세손 정보는 저장 X)
        save_name = f"{final_gloss_id}_{current_idx:03d}_{base_name}.json"

        output_data = {
            "metadata": {
                "gloss_id": final_gloss_id,
                "animation_id": int(gloss_anim_map[final_gloss_id]),
                "original_video": base_name,
                "total_frames": len(clip["frames"])
            },
            "frames": clip["frames"]
        }
        with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=4, ensure_ascii=False)

    return current_max_id, new_gloss_count    


def process_and_save_nms(anno_data, final_data, base_name, nms_counter):
    extracted_nms = sliceGloss.slice_nms(anno_data, final_data)

    for clip in extracted_nms:
        nms_type = clip["nms_type"]

        nms_counter[nms_type] = nms_counter.get(nms_type, 0) + 1
        current_idx = nms_counter[nms_type]

        save_dir = os.path.join(DIR_NMS_JSON, nms_type)
        os.makedirs(save_dir, exist_ok=True)
        save_name = f"{nms_type}_{current_idx:03d}_{base_name}.json"

        output_data = {
            "metadata": {
                "nms_type": nms_type,
                "original_video": base_name,
                "total_frames": len(clip["frames"])
            },
            "frames": clip["frames"]
        }

        with open(os.path.join(save_dir, save_name), 'w', encoding='utf-8') as out_f:
            json.dump(output_data, out_f, indent=4, ensure_ascii=False)

def run_pipeline():
    print(f"🔍 '{EXCEL_MAPPING_PATH}' 엑셀 파일 스캔 중...")

    gloss_anim_map = load_excel_mapping()
    '''
    load_excel_mapping return 값
    {
        "감기1": 1,
        "갑상선1": 2,
        "갑자기1": 3,
        "강하다1": 4,
        "걱정1": 5,
        "검사1": 6,
        "검정1": 7,
        # ... (중략) ...
        "다니다1": 20
    }
    '''
    if not gloss_anim_map:
        return
    
    # AnimationID의 최대값 확인
    current_max_id = max(gloss_anim_map.values())

    # 원본 타겟 단어 리스트 (방향이 없는 기본 형태)
    base_target_glosses = list(gloss_anim_map.keys())
    print(f"✅ 엑셀에서 타겟 단어 {len(base_target_glosses)}개 로드 완료. (현재 최대 ID: {current_max_id})")

    video_files = []
    for root, dirs, files in os.walk(DIR_VIDEOS):
        for file in files:
            if file.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, file))
                # C:/project/dataset/raw_videos/02_NIKL/VXPAKOKS230848240.mp4 (영상 파일의 전체 경로가 video_files에 더해져있음)

    total_files = len(video_files)
    if total_files == 0:
        print("⚠️ 처리할 MP4 영상이 없습니다.")
        return
    
    # 새로 추가될 인칭동사의 개수
    new_gloss_count = 0 

    print(f"🚀 총 {total_files}개의 영상을 찾았습니다. 파이프라인을 가동합니다.\n")

    # 메인 순회 루프
    for index, video_path in enumerate(video_files, start=1):        
        base_name = os.path.splitext(os.path.basename(video_path))[0] # .mp4 제거한 영상 파일 제목
        video_dir = os.path.dirname(video_path) # 파일명을 제외한, 그 파일이 들어있는 디렉토리까지만 추출
        rel_dir = os.path.relpath(video_dir, DIR_VIDEOS) # 하위 폴더 이름만을 알아냄

        # 현재 처리 중인 파일의 진행 상태를 출력합니다.
        print(f"▶️ [{index}/{total_files}] 분석 중: {rel_dir}/{base_name}")

        anno_json_path = os.path.join(video_dir, f"{base_name}.json")
        final_json_path = os.path.join(DIR_FINAL_JSON, rel_dir, f"{base_name}_keypoints_final.json")

        if not os.path.exists(anno_json_path):
            print(f"  -> ⚠️ 정답지(JSON) 없음. 건너뜁니다.")
            continue
        if not os.path.exists(final_json_path):
            print(f"  -> ⚠️ 추출된 뼈대 데이터(Final JSON) 없음. 건너뜁니다.")
            continue
        
        with open(anno_json_path, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)
        with open(final_json_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        # 제스처 추출 및 저장 (액셀 업데이트 포함)
        gloss_counter = {} 
        nms_counter = {} 

        current_max_id, new_gloss_count = process_and_save_gesture(
            anno_data, final_data, base_target_glosses, base_name, gloss_anim_map, 
            current_max_id, new_gloss_count, gloss_counter
        )

        process_and_save_nms(anno_data, final_data, base_name, nms_counter)

    print("\n" + "="*40)

    print("🎉 파이프라인 데이터 구축 완료!")
    if new_gloss_count > 0:
        print(f"✨ 방향(Direction) 정보가 추가되어 엑셀이 갱신되었습니다: +{new_gloss_count}개")
        save_excel_mapping(gloss_anim_map)
    else:
        print("📊 방향 파생 단어가 발견되지 않아 기존 엑셀을 유지합니다.")
    print("="*40)


if __name__ == "__main__":
    run_pipeline()