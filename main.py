import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import leastsq


# MediaPipe Handsの初期化
def initialize_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
    )
    return hands


# 円フィッティング用関数
def calc_circle_fitting(x, y):
    def residuals(p, x, y):
        xc, yc, r = p
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

    x0, y0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))
    p0 = [x0, y0, r0]
    plsq = leastsq(residuals, p0, args=(x, y))
    return plsq[0]  # xc, yc, r


# 改良された真円度の計算関数
def calculate_improved_roundness(x, y, xc, yc, r):
    # 中心からの距離を計算
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    mean_distance = np.mean(distances)

    # 距離の誤差を計算
    distance_error = np.abs(distances - r)
    mean_error = np.mean(distance_error)

    # 滑らかさを計算 (連続する点間の角度変化を評価)
    angles = np.arctan2(y - yc, x - xc)
    angle_diffs = np.diff(np.unwrap(angles))
    smoothness = np.std(angle_diffs)  # 標準偏差が小さいほど滑らか

    # 真円度の総合評価
    roundness = 100 - ((mean_error / r) * 50 + smoothness * 50)
    return max(0, roundness)  # 真円度は0%以上


# 真円度をファイルに記録
def log_roundness(roundness):
    with open("roundness_log.txt", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}, {roundness:.2f}\n")


# 軌跡をフィルタリングする関数
def filter_trajectory(trajectory):
    if len(trajectory) < 2:
        return trajectory

    # 距離が短すぎる点を除外する
    filtered_trajectory = [trajectory[0]]
    for i in range(1, len(trajectory)):
        prev_x, prev_y = filtered_trajectory[-1]
        curr_x, curr_y = trajectory[i]
        distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
        if distance > 5:  # しきい値を適宜調整可能
            filtered_trajectory.append((curr_x, curr_y))
    return filtered_trajectory


# メイン関数
def main():
    hands = initialize_mediapipe_hands()

    # Webカメラの起動
    cap = cv2.VideoCapture(0)
    trajectory = []
    last_seen = time.time()
    hand_detected = True
    last_roundness = None
    last_trajectory = []  # 記録していないときの軌跡保持用
    recording = False  # 記録状態の管理

    # 描画ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()

        # 記録の開始・停止トリガー（スペースキー）
        if cv2.waitKey(1) & 0xFF == ord(" "):
            recording = not recording
            if not recording:
                last_trajectory = trajectory.copy()  # 記録停止時に軌跡を保存
                trajectory = []  # 軌跡をリセット

        if results.multi_hand_landmarks:
            hand_detected = True
            last_seen = current_time

            for hand_landmarks in results.multi_hand_landmarks:
                # 人差し指の指先座標を取得（8番目のランドマーク）
                finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(finger_tip.x * w), int(finger_tip.y * h)

                if recording:
                    # 軌跡を保存
                    trajectory.append((x, y))

                    # 軌跡をフィルタリング
                    trajectory = filter_trajectory(trajectory)

                    # 軌跡を描画
                    for i in range(1, len(trajectory)):
                        cv2.line(
                            frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2
                        )

                    # 円フィッティングと改良された真円度の計算
                    if len(trajectory) > 10:
                        x_vals = np.array([p[0] for p in trajectory])
                        y_vals = np.array([p[1] for p in trajectory])
                        xc, yc, r = calc_circle_fitting(x_vals, y_vals)
                        last_roundness = calculate_improved_roundness(
                            x_vals, y_vals, xc, yc, r
                        )

        # 軌跡と真円度を記録停止中も表示
        if not recording and last_trajectory:
            for i in range(1, len(last_trajectory)):
                cv2.line(
                    frame, last_trajectory[i - 1], last_trajectory[i], (0, 255, 0), 2
                )

        if last_roundness is not None:
            cv2.putText(
                frame,
                f"Roundness: {last_roundness:.2f}%",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # 3秒後にリセット
        if not hand_detected and (current_time - last_seen > 3):
            if len(trajectory) > 10 and recording:
                x_vals = np.array([p[0] for p in trajectory])
                y_vals = np.array([p[1] for p in trajectory])
                xc, yc, r = calc_circle_fitting(x_vals, y_vals)
                last_roundness = calculate_improved_roundness(x_vals, y_vals, xc, yc, r)

                log_roundness(last_roundness)
                trajectory = []  # 軌跡をリセット

        # 記録状態を表示
        status_text = "Recording: ON" if recording else "Recording: OFF"
        cv2.putText(
            frame,
            status_text,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if recording else (0, 0, 255),
            2,
        )

        cv2.imshow("Circle Drawing Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
