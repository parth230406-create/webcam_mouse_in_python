# webcam_mouse_in_python
A computer vision–based virtual mouse that lets you control your cursor using head (nose) movement and perform left/right mouse clicks using eye blinks. Built with OpenCV, MediaPipe Face Mesh, and PyAutoGUI for hands-free human–computer interaction and accessibility use cases.

# Webcam Mouse Control using Face & Eye Tracking 👀🖱️

This project implements a **hands-free virtual mouse system** using a webcam.  
You can move the mouse cursor by moving your head (nose tracking) and perform:

- **Left click** by blinking your **left eye**
- **Right click** by blinking your **right eye**

It uses **MediaPipe Face Mesh** for facial landmark detection, **OpenCV** for video processing, and **PyAutoGUI** for controlling the mouse.

---

## ✨ Features

- Real-time face landmark tracking
- Cursor movement using nose position
- Left and right mouse clicks via eye blinks
- Adjustable sensitivity and dead zones
- Blink debounce to prevent accidental multiple clicks
- Visual debugging (eye landmarks, nose points, EAR values)

---

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV**
- **MediaPipe**
- **PyAutoGUI**
- **Math / Time modules**

---

## 📦 Installation

1. Clone the repository: git clone

2. https://github.com/your-username/webcam-mouse-control.git
   cd webcam-mouse-control

3. Install required dependencies: pip install opencv-python mediapipe pyautogui

4. How to run? --> python webcam_mouse_control.py

5. Controls
Action	Gesture
Move Cursor	Move your head / nose
Left Click	Blink left eye
Right Click	Blink right eye
Exit Program	Press q
