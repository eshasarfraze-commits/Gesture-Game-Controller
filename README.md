# Gesture-Game-Controller

Gesture-Game-Controller is an AI-powered, real-time gesture-based controller for racing games built using computer vision. It allows players to control steering, acceleration, and in-game actions using natural hand movements captured through a webcam.

The system leverages MediaPipe Hands for hand tracking and OpenCV for real-time video processing, translating hand rotation, hand height, and predefined gestures into keyboard inputs. It is optimized for smooth, low-latency gameplay and works well with keyboard-based racing games such as Trackmania.

## Features

- Steering control using hand rotation (tilt)
- Acceleration control using hand height
- Gesture-based actions for gameplay commands
- Real-time processing with ~60 FPS
- Low latency (<50 ms) response
- On-screen visual feedback for hand tracking and control state
- Simple, single-file Python implementation

## Requirements

- Python 3.8 or higher  
- Webcam  
- Supported operating system (Windows recommended for keyboard input)

## Installation

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
requirements.txt includes:

opencv-python
mediapipe
numpy
pydirectinput (optional; install for better game input)
pyautogui (fallback)

## Controls 
Rotate Hand = steer
Hand Height = speed
Fist = brake
Thumbs Up = boost (tap)
Peace = handbrake
OK sign = reset (tap)
Keyboard adjustments at runtime: +/- sensitivity, d/D deadzone, t/T threshold, s save config, ESC exit
