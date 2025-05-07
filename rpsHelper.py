def classify_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]
    open_fingers = 0
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            open_fingers += 1
    if open_fingers == 0:
        return "Rock"
    elif open_fingers in [2, 3]:
        return "Scissors"
    elif open_fingers >= 4:
        return "Paper"
    return "Unknown"

def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Scissors" and computer == "Paper") or \
         (player == "Paper" and computer == "Rock"):
        return "Player Wins!"
    else:
        return "Computer Wins!"