import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --- Configuration Constants ---
# Eye Aspect Ratio (EAR) thresholds for blink detection
# When EAR falls below this, a blink is detected.
LEFT_EYE_BLINK_THRESHOLD = 0.25
RIGHT_EYE_BLINK_THRESHOLD = 0.25

# Time (in seconds) to wait after a click before another click can be registered.
# This prevents multiple clicks from a single prolonged blink.
BLINK_DEBOUNCE_TIME = 0.4

# Sensitivity for cursor movement. Higher values mean faster cursor movement.
CURSOR_SENSITIVITY = 1.5

# Dead zone for nose movement (percentage of screen width/height).
# If nose movement is within this zone, cursor won't move, preventing jitter.
NOSE_DEAD_ZONE_X = 0.1  # 10% of screen width
NOSE_DEAD_ZONE_Y = 0.1  # 10% of screen height

# --- Global Variables for State Management ---
left_eye_closed_start_time = 0
right_eye_closed_start_time = 0
left_click_cooldown_active = False
right_click_cooldown_active = False

# Get screen dimensions for cursor movement mapping
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# --- Helper Function: Calculate Eye Aspect Ratio (EAR) ---
def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given eye.
    EAR is a ratio of distances between eye landmarks, used to detect blinks.
    A smaller EAR indicates a more closed eye.

    Args:
        eye_landmarks (list): A list of 6 (x, y) coordinates representing the eye
                              landmarks from MediaPipe's Face Mesh model.
                              Specific indices for MediaPipe:
                              P1: Horizontal left (index 0)
                              P2: Vertical top-left (index 1)
                              P3: Vertical top-right (index 2)
                              P4: Horizontal right (index 3)
                              P5: Vertical bottom-right (index 4)
                              P6: Vertical bottom-left (index 5)
    Returns:
        float: The calculated Eye Aspect Ratio.
    """
    # Extract coordinates for the 6 key eye landmarks
    p1 = eye_landmarks[0] # Horizontal left
    p2 = eye_landmarks[1] # Vertical top-left
    p3 = eye_landmarks[2] # Vertical top-right
    p4 = eye_landmarks[3] # Horizontal right
    p5 = eye_landmarks[4] # Vertical bottom-right
    p6 = eye_landmarks[5] # Vertical bottom-left

    # Calculate Euclidean distances between the vertical eye landmarks
    # ||P2 - P6||: Distance between vertical top-left and vertical bottom-left
    # ||P3 - P5||: Distance between vertical top-right and vertical bottom-right
    vertical_dist_1 = math.dist(p2, p6)
    vertical_dist_2 = math.dist(p3, p5)

    # Calculate Euclidean distance between the horizontal eye landmarks
    # ||P1 - P4||: Distance between horizontal left and horizontal right
    horizontal_dist = math.dist(p1, p4)

    # Compute EAR
    # Add a small epsilon to the denominator to prevent division by zero
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist + 1e-6)
    return ear

# --- Main Application Logic ---
def run_webcam_mouse_control():
    """
    Initializes the webcam, MediaPipe Face Mesh, and controls the mouse
    based on face/eye movements.
    """
    global left_eye_closed_start_time, right_eye_closed_start_time
    global left_click_cooldown_active, right_click_cooldown_active

    # Initialize MediaPipe Face Mesh
    # static_image_mode=False: For video stream processing
    # max_num_faces=1: Detect only one face
    # min_detection_confidence: Minimum confidence for face detection
    # min_tracking_confidence: Minimum confidence for landmark tracking
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get initial frame dimensions for relative nose position calculation
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame.")
        return
    frame_height, frame_width, _ = frame.shape

    # Define the indices for left and right eye landmarks from MediaPipe Face Mesh
    # These indices are specific to the MediaPipe Face Mesh model
    # Left eye indices (approximate, based on common usage)
    # You might need to adjust these slightly based on exact MediaPipe version or desired points
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] # P1, P2, P3, P4, P5, P6
    # Right eye indices
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144] # P1, P2, P3, P4, P5, P6

    # Get initial nose position to use as a reference point
    # This will be updated once a face is detected
    initial_nose_x, initial_nose_y = frame_width // 2, frame_height // 2
    nose_reference_set = False

    print("Webcam mouse control started. Move your face/nose to move the cursor.")
    print("Blink your left eye for a left click.")
    print("Blink your right eye for a right click.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more intuitive mirror-like view
        frame = cv2.flip(frame, 1)
        # Convert the BGR image to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Set initial nose reference point if not set yet
                if not nose_reference_set:
                    # Get the nose tip landmark (index 1 from MediaPipe)
                    # Convert normalized coordinates to pixel coordinates
                    nose_tip = face_landmarks.landmark[1]
                    initial_nose_x = int(nose_tip.x * frame_width)
                    initial_nose_y = int(nose_tip.y * frame_height)
                    nose_reference_set = True

                # --- Cursor Movement based on Nose ---
                # Get the nose tip landmark (index 1)
                nose_tip = face_landmarks.landmark[1]
                current_nose_x = int(nose_tip.x * frame_width)
                current_nose_y = int(nose_tip.y * frame_height)

                # Calculate the deviation from the initial nose position
                delta_x = current_nose_x - initial_nose_x
                delta_y = current_nose_y - initial_nose_y

                # Apply dead zone: if movement is small, don't move cursor
                if abs(delta_x) > frame_width * NOSE_DEAD_ZONE_X / 2:
                    # Map delta_x to screen movement, scaled by sensitivity
                    move_x = int(delta_x * CURSOR_SENSITIVITY)
                    # Move cursor relatively
                    pyautogui.move(move_x, 0)

                if abs(delta_y) > frame_height * NOSE_DEAD_ZONE_Y / 2:
                    # Map delta_y to screen movement, scaled by sensitivity
                    move_y = int(delta_y * CURSOR_SENSITIVITY)
                    # Move cursor relatively
                    pyautogui.move(0, move_y)

                # Draw a circle at the nose tip for visualization
                cv2.circle(frame, (current_nose_x, current_nose_y), 5, (0, 255, 0), -1)
                # Draw a circle at the initial nose reference point
                cv2.circle(frame, (initial_nose_x, initial_nose_y), 5, (0, 0, 255), -1)

                # --- Eye Blink Detection and Clicks ---
                landmarks = face_landmarks.landmark
                left_eye_points = []
                right_eye_points = []

                # Extract pixel coordinates for left eye landmarks
                for idx in LEFT_EYE_INDICES:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    left_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1) # Draw left eye points

                # Extract pixel coordinates for right eye landmarks
                for idx in RIGHT_EYE_INDICES:
                    x = int(landmarks[idx].x * frame_width)
                    y = int(landmarks[idx].y * frame_height)
                    right_eye_points.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1) # Draw right eye points

                if len(left_eye_points) == 6:
                    left_ear = calculate_ear(left_eye_points)
                    cv2.putText(frame, f"L_EAR: {left_ear:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    current_time = time.time()

                    # Left Eye Blink Detection
                    if left_ear < LEFT_EYE_BLINK_THRESHOLD:
                        if not left_click_cooldown_active:
                            if left_eye_closed_start_time == 0: # First frame of closure
                                left_eye_closed_start_time = current_time
                            # If eye has been closed for a sufficient duration (e.g., a real blink)
                            # You can add a duration check here if needed, but EAR threshold is often enough
                            # For simple blink detection, just check if it's below threshold
                            pyautogui.click()
                            print("Left Click Detected!")
                            left_click_cooldown_active = True
                            left_eye_closed_start_time = 0 # Reset for next blink
                    else: # Eye is open
                        if left_click_cooldown_active and (current_time - left_eye_closed_start_time > BLINK_DEBOUNCE_TIME):
                            left_click_cooldown_active = False
                        left_eye_closed_start_time = 0 # Reset if eye opens before cooldown

                if len(right_eye_points) == 6:
                    right_ear = calculate_ear(right_eye_points)
                    cv2.putText(frame, f"R_EAR: {right_ear:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    current_time = time.time()

                    # Right Eye Blink Detection
                    if right_ear < RIGHT_EYE_BLINK_THRESHOLD:
                        if not right_click_cooldown_active:
                            if right_eye_closed_start_time == 0: # First frame of closure
                                right_eye_closed_start_time = current_time
                            pyautogui.rightClick()
                            print("Right Click Detected!")
                            right_click_cooldown_active = True
                            right_eye_closed_start_time = 0 # Reset for next blink
                    else: # Eye is open
                        if right_click_cooldown_active and (current_time - right_eye_closed_start_time > BLINK_DEBOUNCE_TIME):
                            right_click_cooldown_active = False
                        right_eye_closed_start_time = 0 # Reset if eye opens before cooldown

                # Optionally draw the face mesh landmarks
                # mp_drawing.draw_landmarks(
                #     image=frame,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())

        # Display the frame
        cv2.imshow('Webcam Mouse Control', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    run_webcam_mouse_control()