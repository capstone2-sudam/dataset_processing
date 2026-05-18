import copy

# gloss 단위 프레임 추출
def slice_gestures(anno_data, final_data, target_glosses):
    all_frames = final_data.get("frames", [])
    if not all_frames:
        return []
    
    extracted_gesture = []
    processed_signatures = set() # 중복 방지용으로 작성 (gloss_id, start, end)
    
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

        if direction:
            src = direction.get("source", "") # 비어있으면 "" 가져옴
            tgt = direction.get("target", "") # 비어있으면 "" 가져옴

            direction_str = f"{src}{tgt}".strip()
            
            # direction_str이 비어있지 않은 경우(즉, 숫자가 하나라도 있는 경우)에만 꼬리표를 붙임
            if direction_str:
                final_gloss_id = f"{base_gloss_id}_{direction_str}"
        
        start_sec = gesture.get("start", 0)
        end_sec = gesture.get("end", 0)

        # |start(end)_strong - start(end)_weak| < 0.1초
        # (gloss_id, 시작, 끝)이 완전히 똑같은 데이터는 우세손 따지지 않고 1번만 처리하기
        sig = (final_gloss_id, start_sec, end_sec)
        if sig in processed_signatures:
            continue
        processed_signatures.add(sig)

        start_ms = start_sec * 1000
        end_ms = end_sec * 1000

        clip_frames = []
        for frm in all_frames:
            timestamp = frm.get("timestamp", 0)
            if start_ms <= timestamp <= end_ms:
                new_frm = copy.deepcopy(frm)

                # gesture이므로 얼굴 관련 데이터를 싹 지움
                new_frm["face_keypoints"] = []
                new_frm["lip_keypoints"] = []
                new_frm["eyebrow_keypoints"] = []
                new_frm["eye_keypoints"] = []

                clip_frames.append(new_frm)

        if clip_frames:
            extracted_gesture.append({
                "gloss_id": final_gloss_id,
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

                    new_frm["pose_keypoints"] = []
                    new_frm["left_hand_keypoints"] = []
                    new_frm["right_hand_keypoints"] = []

                    clip_frames.append(new_frm)
            
            if clip_frames:
                extracted_nms.append({
                    "nms_type": nms_type,
                    "frames": clip_frames
                })
    
    return extracted_nms