import time
import json
import threading
from collections import deque
from pathlib import Path
import math

import cv2
import numpy as np
import mediapipe as mp


try:
    import pydirectinput as input_lib
    INPUT_LIB = "pydirectinput"
except Exception:
    import pyautogui as input_lib
    INPUT_LIB = "pyautogui"

# ----------------------------
# Config / Tunables (OPTIMIZED FOR TRACKMANIA)
# ----------------------------
CONFIG_FILE = "gesture_config_trackmania.json" 

TARGET_FPS = 60  
CAM_WIDTH = 640
CAM_HEIGHT = 480

# --- STEERING CONFIG (SIMPLIFIED) ---
STEER_DEADZONE = 0.12       
STEER_THRESHOLD = 0.20      
STEER_SENSITIVITY = 0.75   
STEER_SMOOTH_ALPHA = 0.70   

# --- ACCELERATION Y-POSITION CONFIG ---
ACCEL_Y_LOW = 0.65         
ACCEL_Y_MID = 0.45          
ACCEL_Y_HIGH = 0.25        
ACCEL_SMOOTH_ALPHA = 0.55   

# --- GESTURE CONFIG ---
GESTURE_CONFIDENCE_FRAMES = 2
GESTURE_DEBOUNCE_TIME = 0.15   

GESTURE_KEYS = {
    "open_palm": "up",      
    "fist": "down",         
    "thumbs_up": "shift",    
    "peace": "space",       
    "ok_sign": "backspace"   
}

# ----------------------------
# Config management
# ----------------------------
DEFAULT_CONFIG = {
    "steer_deadzone": STEER_DEADZONE,
    "steer_threshold": STEER_THRESHOLD,
    "steer_sensitivity": STEER_SENSITIVITY,
    "steer_smooth_alpha": STEER_SMOOTH_ALPHA,
    "gesture_confidence_frames": GESTURE_CONFIDENCE_FRAMES,
    "accel_y_low": ACCEL_Y_LOW,
    "accel_y_mid": ACCEL_Y_MID,
    "accel_y_high": ACCEL_Y_HIGH,
    "accel_smooth_alpha": ACCEL_SMOOTH_ALPHA
}

def load_config():
    p = Path(CONFIG_FILE)
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
            DEFAULT_CONFIG.update(cfg)
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        Path(CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass

config = load_config()

# ----------------------------
# Input abstraction
# ----------------------------
class InputSender:
    def __init__(self):
        self.holding = set()
        self.lib = input_lib
        self.using = INPUT_LIB
        self.last_change = {}  

    def key_down(self, key):
        now = time.time()
        if key in self.holding:
            return  
        
        
        if key in self.last_change and (now - self.last_change[key]) < 0.05:
            return
            
        if self.using == "pydirectinput":
            self.lib.keyDown(key)
        else:
            self.lib.keyDown(key)
        self.holding.add(key)
        self.last_change[key] = now

    def key_up(self, key):
        now = time.time()
        if key not in self.holding:
            return  
            
        # Prevent rapid toggling
        if key in self.last_change and (now - self.last_change[key]) < 0.05:
            return
            
        if self.using == "pydirectinput":
            self.lib.keyUp(key)
        else:
            self.lib.keyUp(key)
        self.holding.discard(key)
        self.last_change[key] = now

    def tap(self, key):
        if self.using == "pydirectinput":
            self.lib.press(key)
        else:
            self.lib.press(key)

    def release_all(self):
        for k in list(self.holding):
            try:
                self.key_up(k)
            except Exception:
                pass
        self.holding.clear()

INPUT = InputSender()

# ----------------------------
# Camera thread
# ----------------------------
class CameraStream:
    def __init__(self, src=0, width=CAM_WIDTH, height=CAM_HEIGHT, target_fps=TARGET_FPS):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        self.lock = threading.Lock()
        self._frame = None
        self._ret = False
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self._ret = ret
                self._frame = frame.copy()
        self.cap.release()

    def read(self):
        with self.lock:
            return self._ret, self._frame.copy() if self._frame is not None else (False, None)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# ----------------------------
# MediaPipe init
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_proc = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.7
)

# ----------------------------
# Smoothing & gesture tracking
# ----------------------------
prev_steer = 0.0
prev_accel_level = 0  
gesture_history = deque(maxlen=5)
last_gesture_time = {}

def ema(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def normalize_lighting(frame):
    """CLAHE brightness/contrast normalization."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Compute steering value from hand rotation angle
def compute_steering(landmarks, cfg):
    """
    Compute steering from hand rotation (tilt angle).
    Returns: steer_value (-1 to 1), rotation_angle (degrees)
    """
    # Use wrist (0) and middle finger MCP (9) to calculate hand orientation
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    
    # Calculate angle of hand rotation
    # dx and dy represent the vector from wrist to middle MCP
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    
    # Calculate angle in degrees (-90 to +90)
    # Negative angle = hand tilted left, Positive = hand tilted right
    angle_rad = math.atan2(dx, -dy)  # Negative dy because Y increases downward
    angle_deg = math.degrees(angle_rad)
    
    # Normalize angle to -1 to +1 range
    # Assume max rotation of ~45 degrees in either direction
    max_angle = 45.0
    steer_raw = np.clip(angle_deg / max_angle, -1.0, 1.0)
    
    # Apply deadzone
    deadzone = cfg.get("steer_deadzone", STEER_DEADZONE)
    if abs(steer_raw) < deadzone:
        steer_val = 0.0
    else:
        # Scale the remaining range
        sensitivity = cfg.get("steer_sensitivity", STEER_SENSITIVITY)
        sign = 1 if steer_raw > 0 else -1
        magnitude = (abs(steer_raw) - deadzone) / (1.0 - deadzone)
        steer_val = sign * magnitude * sensitivity
    
    # Clamp to valid range
    steer_val = float(np.clip(steer_val, -1.0, 1.0))
    
    return steer_val, angle_deg

# Compute acceleration level from hand Y position
def compute_acceleration_level(landmarks, cfg):
    """
    Compute acceleration level (0=none, 1=low, 2=mid, 3=high) based on hand Y position.
    Returns: accel_level (0-3), raw_y_position (0 to 1)
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    center_y = (wrist.y + index_mcp.y) / 2.0  
    
    # Get thresholds from config
    y_high = cfg.get("accel_y_high", ACCEL_Y_HIGH)    
    y_mid = cfg.get("accel_y_mid", ACCEL_Y_MID)       
    y_low = cfg.get("accel_y_low", ACCEL_Y_LOW)      
    
    # Determine acceleration level based on Y position
    if center_y <= y_high:
        accel_level = 3  
    elif center_y <= y_mid:
        accel_level = 2  
    elif center_y <= y_low:
        accel_level = 1  
    else:
        accel_level = 0  
    
    return accel_level, center_y

# Improved gesture detection with confidence scoring
def detect_gesture(landmarks):
    """
    Detect gesture with relaxed detection for open palm.
    Returns: (gesture_name, confidence_score)
    """
    tips_idx = [4, 8, 12, 16, 20]
    fingers_open = []
    
    # Check finger states with more lenient threshold
    for i in range(1, 5):
        tip_y = landmarks[tips_idx[i]].y
        pip_y = landmarks[tips_idx[i] - 2].y
        is_open = tip_y < (pip_y + 0.02)  
        fingers_open.append(is_open)
    
    # Thumb check (X-axis) - more lenient
    thumb_x_diff = abs(landmarks[4].x - landmarks[3].x)
    thumb_open = thumb_x_diff > 0.04  # Reduced threshold
    
    # Calculate distances for OK sign - VERY STRICT to prevent false resets
    idx_tip = np.array([landmarks[8].x, landmarks[8].y])
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    tip_distance = np.linalg.norm(idx_tip - thumb_tip)
    
    # Additional check: middle finger should be far from thumb for OK sign
    middle_tip = np.array([landmarks[12].x, landmarks[12].y])
    middle_to_thumb = np.linalg.norm(middle_tip - thumb_tip)
    
    # Gesture detection with confidence
    open_count = sum(fingers_open)
    
    # OK SIGN - VERY STRICT: thumb and index must be touching, other fingers extended
    # This prevents accidental resets during gesture transitions
    if (tip_distance < 0.04 and             
        middle_to_thumb > 0.08 and           
        open_count >= 2):                     
        return "ok_sign", 0.85
    
    # FIST - All fingers closed (BRAKE)
    if open_count == 0 and not thumb_open:
        return "fist", 0.95
    
    # PEACE - Index and middle (HANDBRAKE)
    if fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]:
        return "peace", 0.90
    
    # THUMBS UP - Only thumb extended (BOOST)
    if open_count == 0 and thumb_open:
        return "thumbs_up", 0.90
    
    # Don't detect any gesture if fingers are in transition
    # This prevents false positives during gesture changes
    return None, 0.0

def get_stable_gesture():
    """
    Return gesture only if it's been consistent for required frames.
    """
    if len(gesture_history) < GESTURE_CONFIDENCE_FRAMES:
        return None
    
    # Count occurrences
    gesture_counts = {}
    for g in gesture_history:
        if g is not None:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
    
    if not gesture_counts:
        return None
    
    # Return gesture if it appears in majority of recent frames
    most_common = max(gesture_counts.items(), key=lambda x: x[1])
    if most_common[1] >= max(1, GESTURE_CONFIDENCE_FRAMES - 1):  
        return most_common[0]
    
    return None

# ----------------------------
# HUD drawing
# ----------------------------
def draw_steer_indicator(frame, x, y, w, h, steer, rotation_angle, deadzone, threshold):
    """Draw steering bar with rotation angle visualization."""
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), -1)
    
    center_x = x + w // 2
    
    # Draw deadzone (gray)
    dz_width = int(w * deadzone / 2)
    cv2.rectangle(frame, (center_x - dz_width, y), (center_x + dz_width, y + h), (80, 80, 80), -1)
    
    # Draw threshold zones (yellow)
    thresh_left = center_x - int(w * threshold / 2)
    thresh_right = center_x + int(w * threshold / 2)
    cv2.line(frame, (thresh_left, y), (thresh_left, y + h), (0, 255, 255), 2)
    cv2.line(frame, (thresh_right, y), (thresh_right, y + h), (0, 255, 255), 2)
    
    # Draw current steering position
    fill_w = int((w // 2) * abs(steer))
    color = (100, 150, 255) if steer < 0 else (100, 255, 150)
    
    if steer < 0:
        cv2.rectangle(frame, (center_x - fill_w, y), (center_x, y + h), color, -1)
    else:
        cv2.rectangle(frame, (center_x, y), (center_x + fill_w, y + h), color, -1)
    
    # Center line
    cv2.line(frame, (center_x, y), (center_x, y + h), (255, 255, 255), 2)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)
    
    # Label with rotation angle
    cv2.putText(frame, f"STEER: {steer:.2f} (Angle: {rotation_angle:.1f}°)", (x, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_accel_indicator(frame, x, y, w, h, accel_level, hand_y, cfg):
    """Draw acceleration level indicator with Y position zones."""
    # Background
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), -1)
    
    # Get Y thresholds
    y_high = cfg.get("accel_y_high", ACCEL_Y_HIGH)
    y_mid = cfg.get("accel_y_mid", ACCEL_Y_MID)
    y_low = cfg.get("accel_y_low", ACCEL_Y_LOW)
    
    # Draw zones (from top to bottom)
    zone_height = h // 4
    
    # High zone (top) - Green
    cv2.rectangle(frame, (x, y), (x+w, y+zone_height), (0, 150, 0), -1)
    cv2.putText(frame, "HIGH", (x+5, y+zone_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Mid zone - Yellow
    cv2.rectangle(frame, (x, y+zone_height), (x+w, y+2*zone_height), (0, 150, 150), -1)
    cv2.putText(frame, "MID", (x+5, y+2*zone_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Low zone - Orange
    cv2.rectangle(frame, (x, y+2*zone_height), (x+w, y+3*zone_height), (0, 100, 200), -1)
    cv2.putText(frame, "LOW", (x+5, y+3*zone_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # No accel zone (bottom) - Red
    cv2.rectangle(frame, (x, y+3*zone_height), (x+w, y+h), (0, 0, 150), -1)
    cv2.putText(frame, "OFF", (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw current hand position indicator
    hand_y_px = int(hand_y * h)
    cv2.line(frame, (x, y+hand_y_px), (x+w, y+hand_y_px), (255, 0, 255), 3)
    
    # Highlight active zone
    if accel_level == 3:
        cv2.rectangle(frame, (x, y), (x+w, y+zone_height), (0, 255, 0), 3)
    elif accel_level == 2:
        cv2.rectangle(frame, (x, y+zone_height), (x+w, y+2*zone_height), (0, 255, 255), 3)
    elif accel_level == 1:
        cv2.rectangle(frame, (x, y+2*zone_height), (x+w, y+3*zone_height), (0, 150, 255), 3)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)
    
    # Label
    level_names = ["OFF", "LOW", "MID", "HIGH"]
    cv2.putText(frame, f"ACCEL: {level_names[accel_level]}", (x, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_gesture_indicator(frame, gesture, confidence, active_keys, accel_level):
    """Draw gesture status with debug info."""
    h, w = frame.shape[:2]
    
    # Background panel
    cv2.rectangle(frame, (20, 20), (w - 20, 180), (30, 30, 30), -1)
    cv2.rectangle(frame, (20, 20), (w - 20, 180), (100, 100, 100), 2)
    
    # Gesture name
    gesture_text = gesture.upper() if gesture else "NO GESTURE"
    color = (0, 255, 0) if gesture else (100, 100, 100)
    cv2.putText(frame, f"GESTURE: {gesture_text}", (40, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Accel level display
    level_names = ["OFF", "LOW", "MID", "HIGH"]
    accel_color = [(100,100,100), (0,150,255), (0,255,255), (0,255,0)][accel_level]
    cv2.putText(frame, f"ACCEL LEVEL: {level_names[accel_level]}", (40, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, accel_color, 2)
    
    # Active keys
    keys_text = f"KEYS: {', '.join(active_keys) if active_keys else 'None'}"
    cv2.putText(frame, keys_text, (40, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Controls hint
    cv2.putText(frame, "Rotate Hand = Steer | Hand Height = Speed | Fist = Brake", (40, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ----------------------------
# Input mapping
# ----------------------------
active_holds = set()

def apply_game_input(steer, accel_level, gesture, cfg):
    """
    Apply steering, acceleration level, and gesture inputs to game.
    Acceleration is now controlled by hand Y position (height).
    """
    desired_holds = set()
    
    # STEERING - Digital (works best for Trackmania)
    threshold = cfg.get("steer_threshold", STEER_THRESHOLD)
    if steer < -threshold:
        desired_holds.add("left")
    elif steer > threshold:
        desired_holds.add("right")
    
    # ACCELERATION - Based on hand height (Y position)
    # Level 1-3 = holding up key with different "pressures" (but keyboard only does on/off)
    # In Trackmania, we'll use: Level 1-3 all hold "up", but visually show the level
    if accel_level >= 1:
        desired_holds.add("up")
    
    # BRAKE - Gesture based (overrides acceleration)
    if gesture == "fist":
        desired_holds.discard("up")  # Remove acceleration if braking
        desired_holds.add("down")
    
    # HANDBRAKE - Gesture based
    if gesture == "peace":
        desired_holds.add("space")
    
    # Apply changes
    global active_holds
    
    # Release keys no longer needed
    for key in active_holds - desired_holds:
        INPUT.key_up(key)
    
    # Press new keys
    for key in desired_holds - active_holds:
        INPUT.key_down(key)
    
    active_holds = desired_holds.copy()
    
    # Discrete taps (with debouncing)
    now = time.time()
    
    # BOOST - Thumbs up
    if gesture == "thumbs_up":
        if gesture not in last_gesture_time or (now - last_gesture_time[gesture]) > GESTURE_DEBOUNCE_TIME:
            INPUT.tap(GESTURE_KEYS.get("thumbs_up", "shift"))
            last_gesture_time[gesture] = now
    
    # RESET - OK sign (with much longer debounce to prevent accidents)
    if gesture == "ok_sign":
        if gesture not in last_gesture_time or (now - last_gesture_time[gesture]) > 1.0:  # 1 second debounce
            INPUT.tap(GESTURE_KEYS.get("ok_sign", "backspace"))
            last_gesture_time[gesture] = now

# ----------------------------
# Main loop
# ----------------------------
def main():
    global prev_steer, prev_accel_level
    
    stream = CameraStream(0)
    cfg = config.copy()
    
    print("=" * 60)
    print("TRACKMANIA GESTURE CONTROLLER")
    print("=" * 60)
    print("Keyboard: [+/-] Sensitivity | [d/D] Deadzone | [s] Save | [ESC] Exit")
    print("=" * 60)
    
    prev_time = time.time()
    frame_times = deque(maxlen=30)
    
    with hands_proc:
        while True:
            t0 = time.time()
            ret, frame = stream.read()
            
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            
            # Flip and enhance
            frame = cv2.flip(frame, 1)
            frame = normalize_lighting(frame)
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_proc.process(rgb)
            
            detected_gesture = None
            steer_val = 0.0
            accel_level = 0
            rotation_angle = 0.0
            hand_y = 0.5
            
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                landmarks = lm.landmark
                
                # Compute steering from rotation
                steer_raw, rotation_angle = compute_steering(landmarks, cfg)
                
                # Apply smoothing
                prev_steer = ema(prev_steer, steer_raw, cfg.get("steer_smooth_alpha", STEER_SMOOTH_ALPHA))
                steer_val = prev_steer
                
                # Compute acceleration level from Y position
                accel_raw, hand_y = compute_acceleration_level(landmarks, cfg)
                
                # Smooth acceleration level changes
                alpha = cfg.get("accel_smooth_alpha", ACCEL_SMOOTH_ALPHA)
                prev_accel_level = ema(prev_accel_level, accel_raw, alpha)
                accel_level = int(round(prev_accel_level))
                accel_level = max(0, min(3, accel_level))  # Clamp to 0-3
                
                # Detect gesture
                gesture, confidence = detect_gesture(landmarks)
                gesture_history.append(gesture)
                detected_gesture = get_stable_gesture()
                
            else:
                # No hand detected - center steering, no acceleration
                prev_steer = ema(prev_steer, 0.0, cfg.get("steer_smooth_alpha", STEER_SMOOTH_ALPHA))
                steer_val = prev_steer
                prev_accel_level = ema(prev_accel_level, 0, cfg.get("accel_smooth_alpha", ACCEL_SMOOTH_ALPHA))
                accel_level = 0
                rotation_angle = 0.0
                gesture_history.append(None)
                detected_gesture = None
            
            # Apply inputs to game
            apply_game_input(steer_val, accel_level, detected_gesture, cfg)
            
            # Draw HUD
            # Steering indicator (bottom left)
            draw_steer_indicator(frame, 20, h - 100, 400, 60, steer_val, rotation_angle,
                               cfg.get("steer_deadzone", STEER_DEADZONE),
                               cfg.get("steer_threshold", STEER_THRESHOLD))
            
            # Acceleration indicator (right side)
            draw_accel_indicator(frame, w - 100, h - 350, 60, 280, accel_level, hand_y, cfg)
            
            # Gesture indicator (top)
            draw_gesture_indicator(frame, detected_gesture, 0.0, active_holds, accel_level)
            
            # FPS counter
            now = time.time()
            frame_times.append(now - prev_time)
            prev_time = now
            avg_fps = len(frame_times) / sum(frame_times) if frame_times else 0
            
            cv2.putText(frame, f"FPS: {int(avg_fps)}", (w - 120, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw rotation angle indicator (visual guide)
            # Draw a small "steering wheel" visualization
            wheel_center_x = w // 2
            wheel_center_y = 60
            wheel_radius = 30
            
            # Draw wheel circle
            cv2.circle(frame, (wheel_center_x, wheel_center_y), wheel_radius, (100, 100, 100), 2)
            
            # Draw rotation line
            angle_rad = math.radians(rotation_angle)
            end_x = int(wheel_center_x + wheel_radius * math.sin(angle_rad))
            end_y = int(wheel_center_y - wheel_radius * math.cos(angle_rad))
            
            line_color = (100, 150, 255) if rotation_angle < 0 else (100, 255, 150)
            cv2.line(frame, (wheel_center_x, wheel_center_y), (end_x, end_y), line_color, 3)
            cv2.circle(frame, (wheel_center_x, wheel_center_y), 5, (255, 255, 255), -1)
            
            # Show frame
            cv2.imshow("Trackmania Gesture Controller", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Keyboard controls
            if key == 27:  # ESC
                break
            elif key == ord('+') or key == ord('='):
                cfg["steer_sensitivity"] = min(2.0, cfg["steer_sensitivity"] + 0.1)
            elif key == ord('-') or key == ord('_'):
                cfg["steer_sensitivity"] = max(0.5, cfg["steer_sensitivity"] - 0.1)
            elif key == ord('d'):
                cfg["steer_deadzone"] = max(0.05, cfg["steer_deadzone"] - 0.02)
            elif key == ord('D'):
                cfg["steer_deadzone"] = min(0.4, cfg["steer_deadzone"] + 0.02)
            elif key == ord('t'):
                cfg["steer_threshold"] = max(0.1, cfg["steer_threshold"] - 0.05)
            elif key == ord('T'):
                cfg["steer_threshold"] = min(0.5, cfg["steer_threshold"] + 0.05)
            elif key == ord('s'):
                save_config(cfg)
            elif key == ord('r'):
                cfg = DEFAULT_CONFIG.copy()
            
            # Frame timing
            elapsed = time.time() - t0
            sleep_time = max(0, (1.0 / TARGET_FPS) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Cleanup
    INPUT.release_all()
    stream.stop()
    cv2.destroyAllWindows()
    save_config(cfg)

if __name__ == "__main__":

    main()