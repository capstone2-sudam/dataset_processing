# ksl_translation
수어 장갑의 sensor data와 태블릿 카메라의 video data를 통합하여 한국 수어 문장 및 단어를 번역

### dataset 구축
1. dataset 폴더 생성 후 dataset 폴더 안에 raw_videos 폴더 생성하기
2. raw_videos에 추출하고자 하는 영상 폴더들 전부 넣기 
(한글 이름의 영상도 추출 가능하는 것 확인 완료 & 경로는 영어 경로로 설정해주기)
3. dataset_pipeline.py 코드 수행
4. 터미널에서 좌표값 추출(1단계) -> 보간 (2단계) -> 정규화 (3단계) -> 우세손 추출 (4단계) 수행 상황 출력됨
(만약 1, 2단계의 json은 저장하고 싶지 않으면 코드에서 표시된 부분 주석처리 하기)
5. 최종 사용하는 데이터는 dominant_hand 딕셔너리에 있는 json 파일들
(metadata에서 "dominant_hand" 부분이 우세손, frames가 추출된 keypoints들)

### keypoints 구조
총 7개 부위에서 추출
- face_keypoints (8개)
- lip_keypoints (40개)
- eyebrow_keypoints (10개)
- eye_keypoints (32개)
- pose_keypoints (9개)
- left_hand_keypoints (21개)
- right_hand_keypoints (21개)