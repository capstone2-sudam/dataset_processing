# ksl_translation (한국 수어 번역)
영상에서 수어 동작의 3D keypoints를 추출하고 딥러닝 모델 학습 및 Unity 애니메이션 매핑을 위한 데이터셋을 구축하는 파이프라인
수어 장갑의 sensor data와 태블릿 카메라의 video data를 통합하여 한국 수어 문장 및 단어 번역(Sign-to-Text)을 수행하는 것을 목표로 한다.

## dataset 구축 가이드
1. 프로젝트 루트 경로에 `dataset` 폴더를 생성하고, 그 안에 `raw_videos` 폴더를 생성한다.
2. `raw_videos` 폴더 안에 추출하고자 하는 원본 영상(`.mp4`)들을 모두 넣는다.
    * 하위 폴더로 구조화되어 있어도 자동으로 스캔 진행
    * 영상 파일명에 한글이 포함되어 있어도 정상적으로 추출함 (단, 상위 디렉토리 경로는 영문 사용 권장)
3. dataset_pipeline.py 코드 수행
4. 진행 상황이 터미널에 출력되며, 다음 3단계가 순차적으로 자동 수행됨
- [Step 1] 좌표값 추출 (Raw)
- [Step 2] 결측치 선형 보간 (Interpolation)
- [Step 3] 정규화 및 최종 패키징 (Normalization & Packaging)
(중간 단계(step 1, 2)의 json 파일을 저장하고 싶지 않으면 dataset_pipeline.py 코드에서 `save_to_json` 코드 부분 주석처리 하기)
5. 파이프라인이 완료되면 `final_dataset` 폴더 내에 생성된 `_final.json` 파일들을 모델 학습 및 Unity 애니메이션 매핑에 사용

## keypoints 구조
MediaPipe를 활용하여 1개 프레임당 총 7개 부위, 141개의 3D 좌표(X, Y, Z)를 추출한다.
- face_keypoints (9개) : 머리 방향 및 뼈대 (코, 귀 등)
- lip_keypoints (40개) : 입술 윤곽 (수어의 비수지 기호 파악)
- eyebrow_keypoints (10개) : 눈썹 윤곽
- eye_keypoints (32개) : 눈꺼풀 및 눈동자
- pose_keypoints (8개) : 상체 뼈대 (어깨, 팔꿈치 등)
- left_hand_keypoints (21개) : 왼손 관절
- right_hand_keypoints (21개) : 오른손 관절

## 최종 저장 JSON 파일 구조
최종 결과물은 메타데이터와 전체 프레임 데이터를 포함하는 아래와 같은 구조를 가진다.

```python
final_package = 
{
    "metadata": {
        "total_frames": 105,
        "video_fileName": "NIA_SL_WORD0001",
        "fps": 30
        },
    "frames": [
        {
            "frame_number": 1,
            "timestamp": 33,
            "face_keypoints": [[0.0, 0.1, -0.3], ...],
            "lip_keypoints": [[0.1, 0.1, 0.0], ...],
            "eyebrow_keypoints": [...],
            "eye_keypoints": [...],
            "pose_keypoints": [...],
            "left_hand_keypoints": [...],
            "right_hand_keypoints": [...],
        },
        {
            "frame_number": 2,
            "timestamp": 66,
            "face_keypoints": [...],
            "..." : "..."
        }
    ]        
}
```
