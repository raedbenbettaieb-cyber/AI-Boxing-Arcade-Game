import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'pose_landmarker_full.task')

if not os.path.exists(model_path):
    import urllib.request
    print("Downloading AI Model file...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, model_path)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

score = 0
high_score = 0
player_action = "IDLE"
last_punch_time = 0
PUNCH_COOLDOWN = 0.35
PUNCH_THRESHOLD = 0.55

right_punch_locked = False
left_punch_locked = False
last_hand_used = None
points_earned = 0

GAME_DURATION = 30.0
start_time = time.time()
game_over = False

bag_tilt = 0.0
bag_target_tilt = 0.0
bag_color = (60, 60, 220)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

cv2.namedWindow('2D AI Boxing Arcade', cv2.WINDOW_NORMAL)
cv2.resizeWindow('2D AI Boxing Arcade', 1600, 900)

with PoseLandmarker.create_from_options(options) as landmarker:
    print("\nPress 'R' to restart when game is over. Press 'Q' to close.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        current_time = time.time()

        if not game_over:
            time_left = GAME_DURATION - (current_time - start_time)
            if time_left <= 0:
                time_left = 0
                game_over = True
                player_action = "GAME OVER"
                if score > high_score:
                    high_score = score
        else:
            time_left = 0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                r_shoulder, r_wrist = landmarks[11], landmarks[15]
                l_shoulder, l_wrist = landmarks[12], landmarks[16]

                r_distance = math.sqrt((r_wrist.x - r_shoulder.x)**2 + (r_wrist.y - r_shoulder.y)**2)
                l_distance = math.sqrt((l_wrist.x - l_shoulder.x)**2 + (l_wrist.y - l_shoulder.y)**2)

                if not game_over:
                    if current_time - last_punch_time > PUNCH_COOLDOWN:
                        player_action = "IDLE"
                        bag_target_tilt = 0.0
                        bag_color = (60, 60, 220)

                    if r_distance > PUNCH_THRESHOLD:
                        player_action = "RIGHT PUNCH"
                        bag_target_tilt = 45.0
                        bag_color = (0, 230, 255)

                        if not right_punch_locked:
                            points_earned = 2 if last_hand_used == "LEFT" else 1
                            score += points_earned
                            last_hand_used = "RIGHT"
                            last_punch_time = current_time
                            right_punch_locked = True
                    else:
                        right_punch_locked = False

                    if l_distance > PUNCH_THRESHOLD:
                        player_action = "LEFT PUNCH"
                        bag_target_tilt = -45.0
                        bag_color = (0, 230, 255)

                        if not left_punch_locked:
                            points_earned = 2 if last_hand_used == "RIGHT" else 1
                            score += points_earned
                            last_hand_used = "LEFT"
                            last_punch_time = current_time
                            left_punch_locked = True
                    else:
                        left_punch_locked = False
        else:
            if game_over:
                bag_target_tilt = 0.0
                bag_color = (100, 100, 100)

        bag_tilt += (bag_target_tilt - bag_tilt) * 0.2

        cv2.rectangle(frame, (0, int(h * 0.85)), (w, h), (40, 40, 40), -1)
        cv2.line(frame, (0, int(h * 0.85)), (w, int(h * 0.85)), (0, 165, 255), 4)

        anchor_x = int(w * 0.75)
        anchor_y = 60
        rope_length = 400

        rad = math.radians(bag_tilt)
        bag_center_x = anchor_x + int(rope_length * math.sin(rad))
        bag_center_y = anchor_y + int(rope_length * math.cos(rad))
        bag_radius = 65

        cv2.line(frame, (anchor_x, anchor_y), (bag_center_x, bag_center_y - 120), (150, 150, 150), 5)
        cv2.rectangle(frame, (bag_center_x - bag_radius, bag_center_y - 120), (bag_center_x + bag_radius, bag_center_y + 80), bag_color, -1)
        cv2.rectangle(frame, (bag_center_x - bag_radius, bag_center_y - 120), (bag_center_x + bag_radius, bag_center_y + 80), (0, 0, 0), 3)
        cv2.circle(frame, (bag_center_x, bag_center_y - 120), bag_radius, bag_color, -1)
        cv2.circle(frame, (bag_center_x, bag_center_y - 120), bag_radius, (0, 0, 0), 3)
        cv2.circle(frame, (bag_center_x, bag_center_y + 80), bag_radius, bag_color, -1)
        cv2.circle(frame, (bag_center_x, bag_center_y + 80), bag_radius, (0, 0, 0), 3)

        if not game_over and result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                for start_idx, end_idx in POSE_CONNECTIONS:
                    s_pt = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    e_pt = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(frame, s_pt, e_pt, (0, 255, 255), 4)

        if game_over:
            text_col = (0, 0, 255)
        elif "RIGHT" in player_action:
            text_col = (0, 0, 255)
        elif "LEFT" in player_action:
            text_col = (255, 0, 0)
        else:
            text_col = (0, 255, 0)

        cv2.putText(frame, f"STATUS: {player_action}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(frame, f"STATUS: {player_action}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, text_col, 3)

        timer_col = (0, 255, 255) if time_left > 10 else (0, 0, 255)
        cv2.putText(frame, f"TIME: {time_left:.1f}s", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(frame, f"TIME: {time_left:.1f}s", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, timer_col, 3)

        cv2.putText(frame, f"SCORE: {score}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(frame, f"SCORE: {score}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

        cv2.putText(frame, f"HI-SCORE: {high_score}", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
        cv2.putText(frame, f"HI-SCORE: {high_score}", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (50, 200, 255), 3)

        if not game_over and player_action != "IDLE":
            hit_msg = f"HIT! +{points_earned}"
            popup_color = (0, 255, 255) if points_earned == 2 else (255, 255, 255)
            cv2.putText(frame, hit_msg, (bag_center_x - 100, bag_center_y - 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
            cv2.putText(frame, hit_msg, (bag_center_x - 100, bag_center_y - 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, popup_color, 3)

        if game_over:
            cv2.rectangle(frame, (int(w*0.2), int(h*0.42)), (int(w*0.8), int(h*0.58)), (0, 0, 0), -1)
            cv2.rectangle(frame, (int(w*0.2), int(h*0.42)), (int(w*0.8), int(h*0.58)), (0, 0, 255), 4)
            cv2.putText(frame, "PRESS 'R' TO RESTART ROUND", (int(w*0.26), int(h*0.52)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)

        cv2.imshow('2D AI Boxing Arcade 1080p', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        if game_over and (key == ord('r') or key == ord('R')):
            score = 0
            start_time = time.time()
            game_over = False
            player_action = "IDLE"
            last_hand_used = None
            bag_target_tilt = 0.0
            bag_color = (60, 60, 220)

cap.release()
cv2.destroyAllWindows()