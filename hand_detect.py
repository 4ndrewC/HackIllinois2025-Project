import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os
# Initialize MediaPipe Hands
def find_hand():
    
    os.environ["CPUINFO_LOG_LEVEL"] = "0"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils #use drawing utility
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1) #define landmark style
    handConStyle = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1) #define connection style

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path='hand_gesture_model.tflite')
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Gesture mapping
    gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]
    width = 640
    height = 480

    # Function to normalize landmarks
    def normalize_landmarks(landmarks):
        # Take the first landmark as the reference point (0, 0)
        base_x, base_y = landmarks[0].x, landmarks[0].y
        normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
        return normalized.flatten()

    # function calculate FPS

    #function find the bounding box

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize drawing utils for landmarks
    mp_drawing = mp.solutions.drawing_utils

    # Start capturing video from the camera
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    prev_fps= 0
    counter = 0
    curr_gest = ""
    while cap.isOpened():
        ret, frame = cap.read()
        servo_theta = 0
        servo_phi = 0
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to find hands
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                # Reshape and prepare input data
                input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(input_details[0]['shape'])
                # Set the input tensor
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get the output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Interpret the results
                predicted_class = np.argmax(output_data)
                gesture_name = gesture_names[predicted_class]
                if gesture_name == curr_gest:
                    counter+=1
                else:
                    counter = 0
                    curr_gest = gesture_name
                if counter >= 1:
                    print('Predicted gesture:', gesture_name)
                    return gesture_name
                # Draw the hand landmarks on the frame
                #mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, handLmsStyle,
                                 #   handConStyle)  # draw landmarks styles
                #brect = calc_bounding_rect(frame, hand_landmarks)  # Calculate the bounding rectangle
                #cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0),
                 #           1)  # Draw the bounding rectangle

                # Display the predicted gesture on the frame
               # cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                  #          cv2.LINE_AA)
        #fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
        #prev_fps = fps
        #cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Display the frame
        # cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()
find_hand()