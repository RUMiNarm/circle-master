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


# 真円度の計算関数
def calculate_roundness(x, y, xc, yc, r):
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    error = np.abs(distances - r)
    roundness = 100 - (np.mean(error) / r) * 100  # 百分率で誤差率を計算
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
    last_fitted_circle = None

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

                # 円フィッティングと真円度の計算
                if len(trajectory) > 10:
                    x_vals = np.array([p[0] for p in trajectory])
                    y_vals = np.array([p[1] for p in trajectory])
                    xc, yc, r = calc_circle_fitting(x_vals, y_vals)
                    roundness = calculate_roundness(x_vals, y_vals, xc, yc, r)

                    # フィッティングされた円を描画
                    last_fitted_circle = (int(xc), int(yc), int(r), roundness)

                    # 真円度を表示
                    cv2.putText(
                        frame,
                        f"Roundness: {roundness:.2f}%",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
        else:
            hand_detected = False

        # 手が画面外に出た場合、最後の円を表示
        if not hand_detected and last_fitted_circle:
            xc, yc, r, roundness = last_fitted_circle
            cv2.circle(frame, (xc, yc), r, (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Roundness: {roundness:.2f}%",
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
                roundness = calculate_roundness(x_vals, y_vals, xc, yc, r)

                log_roundness(roundness)
                trajectory = []  # 軌跡をリセット
                last_fitted_circle = None

        cv2.imshow("Circle Drawing Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
