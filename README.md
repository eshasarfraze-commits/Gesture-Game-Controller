#Gesture-Game-Controller

Gesture-Game-Controller is an AI-powered, real-time gesture-based controller for racing games built using computer vision. It allows players to control steering, acceleration, and in-game actions using natural hand movements captured through a webcam.

The system leverages MediaPipe Hands for hand tracking and OpenCV for real-time video processing, translating hand rotation, hand height, and predefined gestures into keyboard inputs. It is optimized for smooth, low-latency gameplay and works well with keyboard-based racing games such as Trackmania.

##Features

Steering control using hand rotation (tilt)

Acceleration control using hand height

Gesture-based actions for gameplay commands

Real-time processing with ~60 FPS

Low latency (<50 ms) response

On-screen visual feedback for hand tracking and control state

Simple, single-file Python implementation

##Requirements

Python 3.8 or higher

Webcam

Supported operating system (Windows recommended for keyboard input)

##Installation

Clone the repository:

git clone https://github.com/your-username/Gesture-Game-Controller.git
cd Gesture-Game-Controller


Install required dependencies:

pip install opencv-python mediapipe numpy pyautogui

##Usage

Ensure your webcam is connected.

Run the controller:

python Gesture-Game-Controller.py


Launch your racing game and start controlling it using hand gestures.

##Notes

Run the script and game with appropriate permissions if inputs are not detected.

Lighting conditions can affect hand tracking accuracy.

This project is intended for experimental, educational, and demonstration purposes.

##License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.
