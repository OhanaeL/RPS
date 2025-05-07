import math
import cv2

def apply_mouth_filter(frame, face_results, mouth_image):

    img_h, img_w = frame.shape[:2]

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]

            left = int(left_mouth.x * img_w), int(left_mouth.y * img_h)
            right = int(right_mouth.x * img_w), int(right_mouth.y * img_h)

            width = int(math.sqrt((right[0] - left[0])**2 + (right[1] - left[1])**2))

            delta_y = right[1] - left[1]
            delta_x = right[0] - left[0]
            angle = -math.degrees(math.atan2(delta_y, delta_x))

            overlay_transparent(frame, mouth_image, left[0], left[1] - width // 4, overlay_size=(width, width // 2), angle=angle)


def apply_eye_filter(frame, face_results, win_left_img, win_right_img):

    if not face_results.multi_face_landmarks:
        return frame

    face_landmarks = face_results.multi_face_landmarks[0]

    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    left = int(left_eye.x * frame.shape[1]), int(left_eye.y * frame.shape[0])
    right = int(right_eye.x * frame.shape[1]), int(right_eye.y * frame.shape[0])

    eye_distance = int(math.hypot(right[0] - left[0], right[1] - left[1]))
    angle = -math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    eye_size = (int(eye_distance * 0.5), int(eye_distance * 0.5))

    frame = overlay_transparent(frame, win_left_img, left[0] - eye_size[0] // 2 + 10, left[1] - eye_size[1] // 2, eye_size, angle)
    frame = overlay_transparent(frame, win_right_img, right[0] - eye_size[0] // 2 - 10, right[1] - eye_size[1] // 2, eye_size, angle)

    return frame

def overlay_transparent(background, overlay, x, y, overlay_size=None, angle=0):

    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

    (h, w) = overlay.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(overlay, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    b, g, r, a = cv2.split(rotated_overlay)
    mask = a / 255.0
    inv_mask = 1.0 - mask

    h, w = rotated_overlay.shape[:2]
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    overlay_clipped = rotated_overlay[0:(y2 - y1), 0:(x2 - x1)]
    mask_clipped = mask[0:(y2 - y1), 0:(x2 - x1)]
    inv_mask_clipped = inv_mask[0:(y2 - y1), 0:(x2 - x1)]

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            background[y1:y2, x1:x2, c] * inv_mask_clipped +
            overlay_clipped[:, :, c] * mask_clipped
        )
    return background