import cv2
import mediapipe as mp
import pyautogui
import util
import time
import numpy as np
from collections import deque
from pynput.mouse import Button, Controller
mouse = Controller()

# Disable PyAutoGUI failsafe for smoother movement
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

# Enhanced smoothing variables
prev_x, prev_y = 0, 0
velocity_x, velocity_y = 0, 0
smooth_factor = 0.92
dpi_multiplier = 2

# Pre-compute center coordinates
CENTER_X = screen_width / 2
CENTER_Y = screen_height / 2
MAX_X = screen_width - 1
MAX_Y = screen_height - 1

# Moving average buffer
coord_history = deque(maxlen=15)

# Pre-compute exponential weights for speed
WEIGHTS = np.array([(i + 1) ** 1.5 for i in range(15)])

# Dead zone and gesture settings
dead_zone = 1
last_click_time = 0
click_delay = 0.2  # Reduced from 0.3 for more responsive clicks
last_gesture = None
gesture_hold_frames = 0
gesture_threshold = 2  # Reduced from 3 for faster detection

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8,  # Increased for smoother tracking
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def move_mouse(index_finger_tip):
    global prev_x, prev_y, coord_history, velocity_x, velocity_y
    
    if index_finger_tip is None:
        return
    
    # Calculate target position with DPI multiplier (vectorized)
    offset_x = (index_finger_tip.x * screen_width - CENTER_X) * dpi_multiplier
    offset_y = (index_finger_tip.y * screen_height - CENTER_Y) * dpi_multiplier
    
    x = max(0, min(MAX_X, int(CENTER_X + offset_x)))
    y = max(0, min(MAX_Y, int(CENTER_Y + offset_y)))
    
    coord_history.append((x, y))
    
    # Fast weighted average using numpy
    history_len = len(coord_history)
    if history_len > 0:
        coords_array = np.array(coord_history)
        weights = WEIGHTS[:history_len]
        avg_x = np.dot(coords_array[:, 0], weights) / weights.sum()
        avg_y = np.dot(coords_array[:, 1], weights) / weights.sum()
    else:
        avg_x, avg_y = x, y
    
    # Initialize on first run
    if prev_x == 0 and prev_y == 0:
        prev_x, prev_y = avg_x, avg_y
        return
    
    # Calculate delta and velocity
    delta_x = avg_x - prev_x
    delta_y = avg_y - prev_y
    
    # Dead zone filter
    if abs(delta_x) < dead_zone and abs(delta_y) < dead_zone:
        return
    
    # Adaptive smoothing based on velocity magnitude
    velocity = (delta_x**2 + delta_y**2) ** 0.5
    adaptive_smooth = 0.98 if velocity < 5 else (0.93 if velocity < 20 else 0.87)
    
    # Apply smoothing
    smooth_x = prev_x + delta_x * adaptive_smooth
    smooth_y = prev_y + delta_y * adaptive_smooth
    
    # Update velocity
    velocity_x = velocity_x * 0.85 + (smooth_x - prev_x) * 0.15
    velocity_y = velocity_y * 0.85 + (smooth_y - prev_y) * 0.15
    
    # Apply momentum and clamp
    final_x = max(0, min(MAX_X, int(smooth_x + velocity_x * 0.1)))
    final_y = max(0, min(MAX_Y, int(smooth_y + velocity_y * 0.1)))
    
    prev_x, prev_y = smooth_x, smooth_y
    pyautogui.moveTo(final_x, final_y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def detect_gesture(frame, landmark_list, processed):
    global last_click_time, last_gesture, gesture_hold_frames
    
    if len(landmark_list) < 21:
        return
    
    # Cache commonly used values
    thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
    current_time = time.time()
    current_gesture = None
    
    # Calculate angles
    index_angle = util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
    middle_angle = util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12])

    # Determine current gesture
    if thumb_index_dist < 50 and index_angle > 90:
        current_gesture = "move"
        move_mouse(find_finger_tip(processed))
        gesture_hold_frames = 0
    elif is_left_click(landmark_list, thumb_index_dist):
        current_gesture = "left_click"
    elif is_right_click(landmark_list, thumb_index_dist):
        current_gesture = "right_click"
    elif is_double_click(landmark_list, thumb_index_dist):
        current_gesture = "double_click"
    
    # Gesture stabilization
    if current_gesture == last_gesture and current_gesture != "move":
        gesture_hold_frames += 1
    else:
        gesture_hold_frames = 0
        last_gesture = current_gesture
    
    # Execute clicks with debouncing
    if gesture_hold_frames >= gesture_threshold and (current_time - last_click_time) > click_delay:
        if current_gesture == "left_click":
            mouse.click(Button.left)
        elif current_gesture == "right_click":
            mouse.click(Button.right)
        elif current_gesture == "double_click":
            pyautogui.doubleClick()
        
        last_click_time = current_time
        gesture_hold_frames = 0


def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = cv2.flip(frame, 1)
            processed = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Extract landmarks efficiently
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            else:
                landmark_list = []

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




