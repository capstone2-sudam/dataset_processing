# 우세손 추출 (의미적 특징 추출)
# 움직임의 분산을 이용
# 전체 dataset_pipeline을 통해 물리적인 특징 추출 후에 추가 수행

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