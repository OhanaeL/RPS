import cv2
import mediapipe as mp
import random
import time
import filterHelper
import rpsHelper

win_left = cv2.imread('assets/images/win_eye.png', cv2.IMREAD_UNCHANGED)
win_right = cv2.imread('assets/images/win_eye2.png', cv2.IMREAD_UNCHANGED)
win_mouth = cv2.imread('assets/images/win_mouth.png', cv2.IMREAD_UNCHANGED)

lose_left = cv2.imread('assets/images/lose_eye.png', cv2.IMREAD_UNCHANGED)
lose_right = cv2.imread('assets/images/lose_eye2.png', cv2.IMREAD_UNCHANGED)
lose_mouth = cv2.imread('assets/images/lose_mouth.png', cv2.IMREAD_UNCHANGED)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

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

cap = cv2.VideoCapture(0)

random_move_interval = 0.1
computer_move_random_timer = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
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
                    result_text = f"You Win!"
                elif winner == "Computer Wins!":
                    computer_score += 1
                    result_text = f"You Lose!"
                else:
                    result_text = f"It's a Draw!"

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

    if winner == "Player Wins!":
        filterHelper.apply_eye_filter(frame, face_results, win_left, win_right)
        filterHelper.apply_mouth_filter(frame, face_results, win_mouth)
    elif winner == "Computer Wins!":
        filterHelper.apply_eye_filter(frame, face_results, lose_left, lose_right)
        filterHelper.apply_mouth_filter(frame, face_results, lose_mouth)

    cv2.rectangle(frame, (0, 0), (w, 115), (20, 20, 20), -1)

    instructions = "'Enter' - Next | 'I' - Landmarks | 'R' - Reset | 'Q' - Quit"
    cv2.putText(frame, instructions, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(frame, f"Round: {round_number} | Player: {player_score} | Computer: {computer_score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    move_text = f"Your Move: {current_move}"
    if not waiting_for_input and current_move in ["Rock", "Paper", "Scissors"]:
        move_text += f" | Hold for {min(3, int(countdown_time) + 1)}s"
    cv2.putText(frame, move_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    if not waiting_for_input:
        if time.time() - computer_move_random_timer >= random_move_interval:
            computer_move_text = f"{random.choice(['Rock', 'Paper', 'Scissors'])}"
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

    cv2.imshow("Rock Paper Scissors - MediaPipe", frame)

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
    elif key == 13 and waiting_for_input:
        round_number += 1
        waiting_for_input = False
        result_text = ""
        prev_gesture = ""

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
