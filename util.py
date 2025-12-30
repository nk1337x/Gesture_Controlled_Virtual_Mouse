import cv2
import mediapipe as mp
import pyautogui
import random
import math
from pynput.mouse import Button, Controller
mouse = Controller()


screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def get_distance(landmark_list):
    """Calculate Euclidean distance between two landmarks"""
    if len(landmark_list) >= 2:
        x1, y1 = landmark_list[0]
        x2, y2 = landmark_list[1]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 1000  # Scale up for easier threshold comparisons
    return 0


def get_angle(point1, point2, point3):
    """Calculate angle at point2 formed by point1-point2-point3"""
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    # Vector from point2 to point1
    v1 = (x1 - x2, y1 - y2)
    # Vector from point2 to point3
    v2 = (x3 - x2, y3 - y2)
    
    # Calculate dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Avoid division by zero
    if magnitude1 * magnitude2 == 0:
        return 0
    
    # Calculate angle in degrees
    angle = math.acos(max(-1, min(1, dot_product / (magnitude1 * magnitude2))))
    return math.degrees(angle)


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        if get_distance([landmark_list[4], landmark_list[5]]) < 50  and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




