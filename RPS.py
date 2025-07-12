import cv2
import mediapipe as mp
import random
import time
import filterHelper
import rpsHelper
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_image(relative_path):
    path = os.path.join(BASE_DIR, relative_path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found or failed to load: {path}")
    return img

def list_available_cameras(max_index=5):
    print("Scanning for available cameras...")
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"Camera {i} is available.")
            available.append(i)
        cap.release()
    return available

available_cameras = list_available_cameras()
if not available_cameras:
    print("No cameras found. Exiting.")
    exit()

try:
    camera_index = int(input(f"Select a camera index from {available_cameras}: "))
    if camera_index not in available_cameras:
        raise ValueError
except ValueError:
    print("Invalid selection. Using default camera index 0.")
    camera_index = 0

cap = cv2.VideoCapture(camera_index)

# Load Filters
win_left = load_image('assets/images/win_eye.png')
win_right = load_image('assets/images/win_eye2.png')
win_mouth = load_image('assets/images/win_mouth.png')

lose_left = load_image('assets/images/lose_eye.png')
lose_right = load_image('assets/images/lose_eye2.png')
lose_mouth = load_image('assets/images/lose_mouth.png')

# Initialize MediaPipe
mp_hands = mp.solutions.hands # type: ignore
mp_drawing = mp.solutions.drawing_utils # type: ignore
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh # type: ignore
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

# Game Variables
show_landmarks = True
player_score = 0
computer_score = 0
player_move = ""
computer_move = ""
winner = ""
result_text = ""
round_number = 1

prev_gesture = ""
gesture_start_time = None
hold_duration = 2
waiting_for_input = False
countdown_time = hold_duration
computer_move_text = "Rock"

random_move_interval = 0.1
computer_move_random_timer = time.time()

# Window setup
fullscreen = False
window_name = "Rock Paper Scissors - MediaPipe"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Game Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_h, original_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(frame_rgb)

    current_move = "No Hand Detected"
    gesture_confirmed = False
    current_time = time.time()

    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            if show_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = rpsHelper.classify_gesture(hand_landmarks.landmark)
            current_move = gesture

            if gesture == prev_gesture:
                if gesture_start_time is None:
                    gesture_start_time = current_time
                elif current_time - gesture_start_time >= hold_duration:
                    gesture_confirmed = True
            else:
                prev_gesture = gesture
                gesture_start_time = current_time

            if gesture_confirmed and not waiting_for_input:
                player_move = gesture
                computer_move = random.choice(["Rock", "Paper", "Scissors"])
                winner = rpsHelper.get_winner(player_move, computer_move)

                if winner == "Player Wins!":
                    player_score += 1
                    result_text = "You Win!"
                elif winner == "Computer Wins!":
                    computer_score += 1
                    result_text = "You Lose!"
                else:
                    result_text = "It's a Draw!"

                waiting_for_input = True
                prev_gesture = ""
                gesture_start_time = None
    else:
        prev_gesture = ""

    if not waiting_for_input:
        if gesture_start_time is not None and not gesture_confirmed:
            countdown_time = max(0, hold_duration - (current_time - gesture_start_time))
        else:
            countdown_time = hold_duration
    else:
        countdown_time = 0

    face_results = face_mesh.process(frame_rgb)

    try:
        if winner == "Player Wins!":
            filterHelper.apply_eye_filter(frame, face_results, win_left, win_right)
            filterHelper.apply_mouth_filter(frame, face_results, win_mouth)
        elif winner == "Computer Wins!":
            filterHelper.apply_eye_filter(frame, face_results, lose_left, lose_right)
            filterHelper.apply_mouth_filter(frame, face_results, lose_mouth)
    except Exception as e:
        print(f"Filter application error: {e}")

    # Overlay UI
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 115), (20, 20, 20), -1)

    instructions = "'Enter' - Next | 'I' - Landmarks | 'R' - Reset | 'F' - Fullscreen | 'Q' - Quit"
    cv2.putText(frame, instructions, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(frame, f"Round: {round_number} | Player: {player_score} | Computer: {computer_score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    move_text = f"Your Move: {current_move}"
    if not waiting_for_input and current_move in ["Rock", "Paper", "Scissors"]:
        move_text += f" | Hold for {min(3, int(countdown_time) + 1)}s"
    cv2.putText(frame, move_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    if not waiting_for_input:
        if time.time() - computer_move_random_timer >= random_move_interval:
            computer_move_text = random.choice(["Rock", "Paper", "Scissors"])
            computer_move_random_timer = time.time()
    else:
        computer_move_text = computer_move

    cv2.putText(frame, f"Computer's Move: {computer_move_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    if result_text:
        text_size, _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_width = text_size[0]
        x_position = (w - text_width) // 2
        y_position = h - 45
        cv2.putText(frame, result_text, (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Show while preserving aspect ratio (no stretching)
    screen_w = 1280
    screen_h = 720
    aspect_ratio = original_w / original_h
    new_w = screen_w
    new_h = int(new_w / aspect_ratio)
    if new_h > screen_h:
        new_h = screen_h
        new_w = int(new_h * aspect_ratio)
    resized_frame = cv2.resize(frame, (new_w, new_h))

    black_canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    y_offset = (screen_h - new_h) // 2
    x_offset = (screen_w - new_w) // 2
    black_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    cv2.imshow(window_name, black_canvas)

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        show_landmarks = not show_landmarks
    elif key == ord('r'):
        player_score = 0
        computer_score = 0
        winner = ""
        player_move = ""
        computer_move = ""
        result_text = ""
        gesture_start_time = None
        prev_gesture = ""
        waiting_for_input = False
    elif key == ord('f'):
        fullscreen = not fullscreen
        prop = cv2.WND_PROP_FULLSCREEN
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, prop, mode)
    elif key == 13 and waiting_for_input:
        round_number += 1
        waiting_for_input = False
        result_text = ""
        prev_gesture = ""

# Cleanup
cap.release()
face_mesh.close()
cv2.destroyAllWindows()
