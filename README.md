# ksl_translation
수어 장갑의 sensor data와 태블릿 카메라의 video data를 통합하여 한국 수어 문장 및 단어를 번역

### dataset 구축
1. raw_videos에 추출하고자 하는 영상 폴더들 전부 넣기 (한글 이름의 영상도 추출 가능하는 것 확인 완료)
2. dataset_pipeline.py 코드 수행
3. 터미널에서 좌표값 추출(1단계) -> 보간 (2단계) -> 정규화 (3단계) 수행 상황 출력됨
   (만약 1, 2단계의 json은 저장하고 싶지 않으면 코드에서 표시된 부분 주석처리 하기)
4. 최종 사용하는 데이터는 normalized 딕셔너리에 있는 json 파일들
5. 이후 normalized 딕셔너리 파일에 있는 json 파일을 우세손 추출에 사용
