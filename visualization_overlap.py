import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# 설정
FPS = 30
COLORS = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 
    'magenta', 'lime', 'brown', 'pink', 'teal', 'navy', 
    'olive', 'maroon', 'gold', 'coral', 'gray', 'indigo', 
    'turquoise', 'crimson'
]

def animate_overlay_comparison(json_paths, output_filename="overlay_comparison.mp4", is_normalized=True):
    """
    여러 개의 수어 영상을 한 화면에 겹쳐서 시각화하는 함수
    - json_paths: JSON 파일 경로들의 리스트
    - is_normalized: 정규화된 데이터인지 여부 (축 범위 설정용)
    """
    all_data = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 🌟 추가된 방어 코드: 데이터가 딕셔너리이고 'frames' 키가 있다면, frames 리스트만 쏙 빼옵니다.
        if isinstance(data, dict) and "frames" in data:
            data = data["frames"]
        
        all_data.append(data)

    # 모든 영상 중 가장 짧은 프레임 수에 맞춤 (동기화)
    total_frames = min(len(d) for d in all_data)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 연결선 정의
    pose_conn = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7)]
    hand_conn = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
    ]

    def update(frame_idx):
        ax.clear()
        
        # 🎯 축 범위 설정
        if is_normalized:
            limit = 1.5  # 정규화 후에는 어깨 중심 기준이므로 범위가 작음
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(limit, -limit)
            # ax.set_xlim(0, 1) # 정규화 전 기본 범위
            # ax.set_ylim(-0.5, 0.5)
            # ax.set_zlim(1, 0)
        else:
            ax.set_xlim(0, 1) # 정규화 전 기본 범위
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(1, 0)
        
        ax.set_title(f"Overlay Comparison - Frame: {frame_idx} / {total_frames}", fontsize=15)

        # 리스트에 있는 모든 영상을 루프 돌며 그리기
        for i, data in enumerate(all_data):
            color = COLORS[i % len(COLORS)]
            frame = data[frame_idx]
            
            pose = np.array(frame.get('pose_keypoints', []))
            face = np.array(frame.get('face_keypoints', []))
            
            # 🎯 [추가할 부분] 입술과 눈썹 데이터도 불러옵니다.
            lips = np.array(frame.get('lip_keypoints', []))
            brows = np.array(frame.get('eyebrow_keypoints', []))
            eyes = np.array(frame.get('eye_keypoints', []))
            
            # 1. Pose (투명도 조절로 겹침 확인)
            if len(pose) > 0:
                ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1], c=color, s=15, alpha=0.6)
                for s, e in pose_conn:
                    if s < len(pose) and e < len(pose):
                        pts = np.array([pose[s], pose[e]])
                        ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], c=color, lw=2, alpha=0.4)

                # 목 연결선 (어깨 중심 -> 코)
                if len(face) > 0:
                    mid_shoulder = (pose[0] + pose[1]) / 2.0
                    neck_pts = np.array([face[0], mid_shoulder])
                    ax.plot(neck_pts[:, 0], neck_pts[:, 2], neck_pts[:, 1], c=color, ls='--', alpha=0.4)

            # 2. Hands
            left_hand = np.array(frame.get('left_hand_keypoints', []))
            right_hand = np.array(frame.get('right_hand_keypoints', []))

            # 왼손과 오른손 배열을 묶어서 반복 처리
            for h_data in [left_hand, right_hand]:
                # 데이터가 존재하고, [0,0,0]으로만 채워진 결측치가 아닐 때만 렌더링
                if len(h_data) >= 21 and not np.all(h_data == 0):
                    # 관절(점) 찍기
                    ax.scatter(h_data[:, 0], h_data[:, 2], h_data[:, 1], c=color, s=8, alpha=0.5)
                    
                    # 뼈대(선) 연결하기
                    for s, e in hand_conn:
                        # 인덱스 에러 방지 안전장치
                        if s < len(h_data) and e < len(h_data):
                            pts = np.array([h_data[s], h_data[e]])
                            ax.plot(pts[:, 0], pts[:, 2], pts[:, 1], c=color, lw=1, alpha=0.3)

            # ==========================================
            # 🎯 [새로 추가된 부분] 3. Face (Lips & Eyebrows)
            # ==========================================
            if len(lips) > 0:
                # 입술 점 찍기 (크기는 작게 s=2)
                ax.scatter(lips[:, 0], lips[:, 2], lips[:, 1], c=color, s=2, alpha=0.6)
            
            if len(brows) > 0:
                # 눈썹 점 찍기
                ax.scatter(brows[:, 0], brows[:, 2], brows[:, 1], c=color, s=2, alpha=0.6)
            
            if len(eyes) > 0:
                # 눈썹 점 찍기
                ax.scatter(eyes[:, 0], eyes[:, 2], eyes[:, 1], c=color, s=2, alpha=0.6)

        # 정면 뷰 고정 (비교를 위해 회전 생략)
        ax.view_init(elev=0, azim=0)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=(1000/FPS), repeat=False)
    
    print(f"🎬 중첩 비교 영상 저장 중: {output_filename}")
    print(f"동영상 변환을 시작합니다... (이 작업은 몇 분 정도 걸릴 수 있습니다.)")
    
    # MP4 저장을 위한 FFMpegWriter 설정
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='AI Lab'), bitrate=1800)
    
    try:
        ani.save(output_filename, writer=writer)
        print(f"✅ 저장 성공! 파일명: {output_filename}")
    except Exception as e:
        print(f"🚨 저장 실패: {e}")
        print("FFmpeg가 설치되어 있지 않을 가능성이 높습니다. 대신 GIF로 저장합니다.")
        # FFmpeg가 없을 경우를 대비한 GIF 자동 저장 Fallback
        ani.save(output_filename.replace('.mp4', '.gif'), writer='pillow', fps=FPS)
        print(f"✅ GIF로 대체 저장되었습니다: {output_filename.replace('.mp4', '.gif')}")

    plt.close(fig) # 창 닫기

# === 사용 예시 ===
paths = [
    r"파일 경로 1",
    r"파일 경로 2"
]
animate_overlay_comparison(paths, "normalized_overlay_2.mp4", is_normalized=True)