import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 💡 뼈대 연결 인덱스 (사용하시는 모델의 Landmark 번호에 맞춰 수정하세요)
# 예: MediaPipe Pose라면 [11, 13], [13, 15] 식으로 연결
POSE_BONES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [3, 5], [0, 6], [1, 7], [6, 7]
]

# 손은 0번부터 20번까지 순서대로 사용하므로 표준을 그대로 사용 가능합니다.
HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4], # 엄지
    [0, 5], [5, 6], [6, 7], [7, 8], # 검지
    [0, 9], [9, 10], [10, 11], [11, 12], # 중지
    [0, 13], [13, 14], [14, 15], [15, 16], # 약지
    [0, 17], [17, 18], [18, 19], [19, 20]  # 소지
]

def draw_points(ax, points, color, size):
    if not points: return
    # 좌표값이 0~1920, 0~1080 수준이므로 리스트 컴프리헨션 사용
    xs = [p[0] for p in points]
    ys = [p[2] for p in points]     # depth
    zs = [-p[1] for p in points]
    ax.scatter(xs, ys, zs, c=color, s=size)

def draw_bones(ax, points, bones, color):
    if not points: return
    for p1_idx, p2_idx in bones:
        # 추출한 인덱스가 유효한지 확인 후 연결
        if p1_idx < len(points) and p2_idx < len(points):
            p1, p2 = points[p1_idx], points[p2_idx]
            ax.plot(
                [p1[0], p2[0]],
                [p1[2], p2[2]],
                [-p1[1], -p2[1]],
                color=color,
                linewidth=2
            )

def run_visualization(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames = data.get("frames", [])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        ax.cla()
        
        # 축 범위를 그에 맞춰 설정해야 합니다.
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-1, 0)

        # 얼굴 확대용 비율
        # ax.set_xlim(0.4, 0.6)
        # ax.set_ylim(-0.2, 0.2)
        # ax.set_zlim(-0.5, -0.2)
        
        # 확대해서 보면 얼굴에서 orange와 black이 따로 노는 걸 볼 수 있는데 이는 face mesh와 pose로 각각 다른 모델로 추출하여 생기는 현상. 
        # 따라서 pose로 추출한 face는 얼굴 회전 각도나 축을 구하는 데 사용하고, face mesh로 추출한 값은 세밀한 표정 조절하는 데 사용 (자세한 사항은 유니티 데이터 전달용.hwp 참고)

        # 라벨 및 타이틀
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame: {frame_idx}')

        ax.set_box_aspect([1, 1, 1])
        # 보이는 각도 설정(정면) - z축 offset은 안 한 상태. unity는 좌표값 기반으로 움직임을 보기 때문에 억지로 맞추면 동작을 보기 어렵다하여 제거함
        ax.view_init(elev=0, azim=-90)

        frame = frames[frame_idx]
        
        # 그리기
        draw_points(ax, frame.get('pose_keypoints', []), 'red', 20)
        draw_bones(ax, frame.get('pose_keypoints', []), POSE_BONES, 'red')

        draw_points(ax, frame.get('left_hand_keypoints', []), 'green', 10)
        draw_bones(ax, frame.get('left_hand_keypoints', []), HAND_BONES, 'green')

        draw_points(ax, frame.get('right_hand_keypoints', []), 'blue', 10)
        draw_bones(ax, frame.get('right_hand_keypoints', []), HAND_BONES, 'blue')

        draw_points(ax, frame.get('face_keypoints', []), 'orange', 5)
        draw_points(ax, frame.get('lip_keypoints', []), 'yellow', 5)
        draw_points(ax, frame.get('eyebrow_keypoints', []), 'purple', 5)
        draw_points(ax, frame.get('eye_keypoints', []), 'black', 5)



    ani = FuncAnimation(fig, update, frames=len(frames), interval=33, repeat=False)
    plt.show()

# 실행
target_file = r"파일 경로"
run_visualization(target_file)