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
    # mean_distance = np.mean(distances)

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


# メイン関数
def main():
    hands = initialize_mediapipe_hands()

    # Webカメラの起動
    cap = cv2.VideoCapture(0)
    trajectory = []
    last_seen = time.time()
    hand_detected = True
    last_roundness = None

    # 描画ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()

        if results.multi_hand_landmarks:
            hand_detected = True
            last_seen = current_time

            for hand_landmarks in results.multi_hand_landmarks:
                # 人差し指の指先座標を取得（8番目のランドマーク）
                finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(finger_tip.x * w), int(finger_tip.y * h)

                # 軌跡を保存
                trajectory.append((x, y))

                # 軌跡を描画
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

                # 円フィッティングと改良された真円度の計算
                if len(trajectory) > 10:
                    x_vals = np.array([p[0] for p in trajectory])
                    y_vals = np.array([p[1] for p in trajectory])
                    xc, yc, r = calc_circle_fitting(x_vals, y_vals)
                    last_roundness = calculate_improved_roundness(
                        x_vals, y_vals, xc, yc, r
                    )

                    # 真円度を表示
                    cv2.putText(
                        frame,
                        f"Roundness: {last_roundness:.2f}%",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
        else:
            hand_detected = False

        # 手が画面外に出た場合も軌跡と真円度を表示
        if not hand_detected and last_roundness is not None:
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)

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
            if len(trajectory) > 10:
                x_vals = np.array([p[0] for p in trajectory])
                y_vals = np.array([p[1] for p in trajectory])
                xc, yc, r = calc_circle_fitting(x_vals, y_vals)
                last_roundness = calculate_improved_roundness(x_vals, y_vals, xc, yc, r)

                log_roundness(last_roundness)
                trajectory = []  # 軌跡をリセット

        cv2.imshow("Circle Drawing Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
