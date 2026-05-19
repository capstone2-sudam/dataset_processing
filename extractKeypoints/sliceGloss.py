import copy

# gloss 단위 프레임 추출
def slice_gestures(anno_data, final_data, target_glosses):
    all_frames = final_data.get("frames", [])
    if not all_frames:
        return []
    
    extracted_gesture = []
    processed_signatures = []
    
    sign_script = anno_data.get("sign_script", {})

    all_gestures = sign_script.get("sign_gestures_strong", []) + sign_script.get("sign_gestures_weak", [])

    for gesture in all_gestures:
        base_gloss_id = gesture.get("gloss_id")

        # target List에 해당 gloss id가 없으면 탐색 중지
        if base_gloss_id not in target_glosses:
            continue

        # 뱡향(Direction) 정보가 있으면 병합하기
        final_gloss_id = base_gloss_id
        direction = gesture.get("direction")
        '''
        저장된 형식
        "direction" : {
            "source": "1",
            "target": "2"    
        }
        '''

        if direction:
            src = direction.get("source", "") # 비어있으면 "" 가져옴
            tgt = direction.get("target", "") # 비어있으면 "" 가져옴

            direction_str = f"{src}{tgt}".strip() # src, tgt가 둘다 ""인 경우 direction_str도 ""
            
            # direction_str이 비어있지 않은 경우(즉, 숫자가 하나라도 있는 경우)에만 꼬리표를 붙임
            if direction_str:
                final_gloss_id = f"{base_gloss_id}_{direction_str}"
        
        start_sec = gesture.get("start", 0) # 예: 2.013
        end_sec = gesture.get("end", 0) # 예: 2.513

        # |start(end)_strong - start(end)_weak| < 0.1초
        is_duplicate = False

        # processed_signatures에 담긴 값을 보고 both인지 아닌 지 확인 후 처리
        for prev_base_id, prev_start, prev_end in processed_signatures:
            # gloss_id가 같고
            if prev_base_id == base_gloss_id:
                # 시작 시간과 끝 시간의 차이가 모두 0.1초 미만인지 확인
                if abs(prev_start - start_sec) < 0.1 and abs(prev_end - end_sec) < 0.1:
                    is_duplicate = True
                    break

        if is_duplicate:
            continue # 중복으로 간주하고 이번 데이터는 건너뜀

        # 새로운 데이터라면 리스트에 기록
        processed_signatures.append((base_gloss_id, start_sec, end_sec))
        start_ms = start_sec * 1000
        end_ms = end_sec * 1000

        clip_frames = []
        for frm in all_frames:
            timestamp = frm.get("timestamp", 0)
            if start_ms <= timestamp <= end_ms:
                new_frm = copy.deepcopy(frm)

                # gesture이므로 얼굴 관련 데이터를 싹 지움
                for key in ["face_keypoints", "lip_keypoints", "eyebrow_keypoints", "eye_keypoints"]:
                    if key in new_frm:
                        del new_frm[key]
                    new_frm[key] = []

                clip_frames.append(new_frm)

        if clip_frames:
            extracted_gesture.append({
                "gloss_id": final_gloss_id,
                "start": start_sec,
                "end": end_sec,
                "frames": clip_frames
            })

    return extracted_gesture    

# 비수지 요소 프레임 추출
def slice_nms(anno_data, final_data):
    all_frames = final_data.get("frames", [])
    if not all_frames:
        return []
    
    extracted_nms = []
    nms_script = anno_data.get("nms_script", {})

    # nms_script 안의 모든 타입(Hno, Mmo, Ebu 등)을 순회
    for nms_type, nms_list in nms_script.items():
        for item in nms_list:
            start_sec = item.get("start", 0)
            end_sec = item.get("end", 0)

            start_ms = start_sec * 1000
            end_ms = end_sec * 1000

            clip_frames = []
            for frm in all_frames:
                timestamp = frm.get("timestamp", 0)
                if start_ms <= timestamp <= end_ms:
                    new_frm = copy.deepcopy(frm)

                    for key in ["left_hand_keypoints", "right_hand_keypoints"]:
                        if key in new_frm:
                            del new_frm[key]
                        new_frm[key] = []

                    original_pose = new_frm.get("pose_keypoints", [])

                    if original_pose and len(original_pose) >= 2:
                        # 이전 추출 코드 기준: 0번이 왼쪽 어깨, 1번이 오른쪽 어깨
                        left_shoulder = original_pose[0]
                        right_shoulder = original_pose[1]
                        del new_frm["pose_keypoints"]
                        new_frm["pose_keypoints"] = [left_shoulder, right_shoulder]
                        
                    else:
                        # 포즈 데이터가 비어있다면 그대로 빈 리스트 유지
                        if "pose_keypoints" in new_frm:
                            del new_frm["pose_keypoints"]
                        new_frm["pose_keypoints"] = []

                    clip_frames.append(new_frm)
            
            if clip_frames:
                extracted_nms.append({
                    "nms_type": nms_type,
                    "start": start_sec,
                    "end": end_sec,
                    "frames": clip_frames
                })
    
    return extracted_nms