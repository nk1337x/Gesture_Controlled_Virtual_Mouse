import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
from collections import deque
from pynput.mouse import Button, Controller

mouse = Controller()

# Disable PyAutoGUI failsafe for smoother movement
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()

# Pre-compute screen center (constant)
SCREEN_CENTER_X = screen_width / 2
SCREEN_CENTER_Y = screen_height / 2
SCREEN_MAX_X = screen_width - 1
SCREEN_MAX_Y = screen_height - 1

# Pre-compute cosine thresholds (avoid acos() every frame)
# 50 degrees = cos(50°) ≈ 0.643, 90 degrees = cos(90°) = 0.0
COS_50_DEG = math.cos(math.radians(50))  # ~0.643
COS_90_DEG = 0.0  # cos(90°) = 0

# Squared distance thresholds (avoid sqrt())
THUMB_INDEX_CLOSE_SQ = 50 * 50  # 2500
THUMB_INDEX_FAR_SQ = 50 * 50    # 2500

# Squared velocity thresholds for adaptive smoothing
VELOCITY_SLOW_SQ = 5 * 5    # 25
VELOCITY_MEDIUM_SQ = 20 * 20  # 400

# Enhanced smoothing variables
prev_x, prev_y = 0, 0
velocity_x, velocity_y = 0, 0
smooth_factor = 0.92  # Ultra-high for maximum smoothness
dpi_multiplier = 1.5  # Increase for faster/more sensitive cursor movement

# Moving average for even smoother tracking
coord_history = deque(maxlen=15)  # Large buffer for ultra-smooth tracking

# Pre-compute weights for moving average (optimization) - NumPy for vectorization
WEIGHTS = np.array([(i + 1) ** 1.5 for i in range(15)], dtype=np.float32)
TOTAL_WEIGHT = np.sum(WEIGHTS)

# Dead zone to filter micro-jitters
dead_zone = 1  # Very small to allow smooth small movements

# Click debouncing
last_click_time = 0
click_delay = 0.3  # 300ms between clicks to prevent accidental multiple clicks
last_gesture = None
gesture_hold_frames = 0
gesture_threshold = 3  # Gesture must be held for 3 frames to trigger

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
    return None


def get_cosine_fast(p1, p2, p3):
    """Calculate cosine of angle at p2 (NO acos() call - just cosine for comparison)"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Vectors from p2
    v1x, v1y = x1 - x2, y1 - y2
    v2x, v2y = x3 - x2, y3 - y2
    
    # Dot product and magnitudes squared (avoid sqrt)
    dot = v1x * v2x + v1y * v2y
    mag1_sq = v1x * v1x + v1y * v1y
    mag2_sq = v2x * v2x + v2y * v2y
    
    if mag1_sq == 0 or mag2_sq == 0:
        return 1.0  # Default to 0 degrees (cos = 1)
    
    # Return cosine directly (no acos!)
    return dot / math.sqrt(mag1_sq * mag2_sq)


def get_distance_squared(p1, p2):
    """Calculate squared distance (NO sqrt() - faster for comparisons)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) * 1000000  # Scale squared


def move_mouse(index_finger_tip):
    global prev_x, prev_y, coord_history, velocity_x, velocity_y
    if index_finger_tip is None:
        return
    
    # Calculate target position with DPI multiplier
    offset_x = (index_finger_tip.x * screen_width - SCREEN_CENTER_X) * dpi_multiplier
    offset_y = (index_finger_tip.y * screen_height - SCREEN_CENTER_Y) * dpi_multiplier
    
    # Clamp to screen boundaries (combine operations)
    x = max(0, min(SCREEN_MAX_X, int(SCREEN_CENTER_X + offset_x)))
    y = max(0, min(SCREEN_MAX_Y, int(SCREEN_CENTER_Y + offset_y)))
    
    # Add to history
    coord_history.append((x, y))
    
    # Calculate weighted moving average with NumPy vectorization
    history_len = len(coord_history)
    if history_len > 0:
        # Convert to NumPy array for vectorized operations
        coords_array = np.array(coord_history, dtype=np.float32)
        weights_slice = WEIGHTS[:history_len]
        
        # Vectorized weighted average (much faster than Python loops)
        avg_x = np.dot(coords_array[:, 0], weights_slice) / np.sum(weights_slice)
        avg_y = np.dot(coords_array[:, 1], weights_slice) / np.sum(weights_slice)
    else:
        avg_x, avg_y = x, y
    
    # Initialize on first run
    if prev_x == 0 and prev_y == 0:
        prev_x, prev_y = avg_x, avg_y
        velocity_x, velocity_y = 0, 0
        return
    
    # Calculate velocity and distance (combined)
    delta_x = avg_x - prev_x
    delta_y = avg_y - prev_y
    
    # Dead zone check (early exit for performance)
    if abs(delta_x) < dead_zone and abs(delta_y) < dead_zone:
        return
    
    # Calculate squared velocity magnitude (NO sqrt!)
    velocity_sq = delta_x * delta_x + delta_y * delta_y
    
    # Adaptive smoothing based on squared velocity
    if velocity_sq < VELOCITY_SLOW_SQ:
        adaptive_smooth = 0.98
    elif velocity_sq < VELOCITY_MEDIUM_SQ:
        adaptive_smooth = 0.93
    else:
        adaptive_smooth = 0.87
    
    # Apply smoothing
    smooth_x = prev_x + delta_x * adaptive_smooth
    smooth_y = prev_y + delta_y * adaptive_smooth
    
    # Update velocity with smoothing
    velocity_x = velocity_x * 0.85 + (smooth_x - prev_x) * 0.15
    velocity_y = velocity_y * 0.85 + (smooth_y - prev_y) * 0.15
    
    # Apply minimal velocity for momentum and clamp
    final_x = max(0, min(SCREEN_MAX_X, int(smooth_x + velocity_x * 0.1)))
    final_y = max(0, min(SCREEN_MAX_Y, int(smooth_y + velocity_y * 0.1)))
    
    # Update previous position
    prev_x, prev_y = smooth_x, smooth_y
    
    # Move mouse
    pyautogui.moveTo(final_x, final_y)


def detect_gesture(frame, landmark_list, processed):
    global last_click_time, last_gesture, gesture_hold_frames
    
    if len(landmark_list) < 21:
        return
    
    # Calculate cosines and squared distances (NO acos, NO sqrt)
    thumb_index_dist_sq = get_distance_squared(landmark_list[4], landmark_list[5])
    
    # Cosine of angles (lower cosine = larger angle)
    index_cos = get_cosine_fast(landmark_list[5], landmark_list[6], landmark_list[8])
    middle_cos = get_cosine_fast(landmark_list[9], landmark_list[10], landmark_list[12])
    
    current_time = time.time()
    current_gesture = None

    # Gesture detection using cosine comparison (inverted logic: smaller angle = larger cosine)
    # Move: thumb-index close AND index straight (small angle = large cosine)
    if thumb_index_dist_sq < THUMB_INDEX_CLOSE_SQ and index_cos < COS_90_DEG:
        current_gesture = "move"
        index_finger_tip = find_finger_tip(processed)
        move_mouse(index_finger_tip)
        gesture_hold_frames = 0
    # Left click: index bent (angle < 50° means cos > cos(50°)) AND middle straight (cos < 0) AND thumb far
    elif index_cos > COS_50_DEG and middle_cos < COS_90_DEG and thumb_index_dist_sq > THUMB_INDEX_FAR_SQ:
        current_gesture = "left_click"
    # Right click: middle bent AND index straight AND thumb far
    elif middle_cos > COS_50_DEG and index_cos < COS_90_DEG and thumb_index_dist_sq > THUMB_INDEX_FAR_SQ:
        current_gesture = "right_click"
    # Double click: both fingers bent AND thumb far
    elif index_cos > COS_50_DEG and middle_cos > COS_50_DEG and thumb_index_dist_sq > THUMB_INDEX_FAR_SQ:
        current_gesture = "double_click"
    
    # Gesture stabilization
    if current_gesture == last_gesture and current_gesture != "move":
        gesture_hold_frames += 1
    else:
        gesture_hold_frames = 0
        last_gesture = current_gesture
    
    # Execute click gestures with debouncing
    if gesture_hold_frames >= gesture_threshold and (current_time - last_click_time) > click_delay:
        if current_gesture == "left_click":
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            last_click_time = current_time
            gesture_hold_frames = 0
        elif current_gesture == "right_click":
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            last_click_time = current_time
            gesture_hold_frames = 0
        elif current_gesture == "double_click":
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            last_click_time = current_time
            gesture_hold_frames = 0


def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            # Extract landmarks efficiently
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()




