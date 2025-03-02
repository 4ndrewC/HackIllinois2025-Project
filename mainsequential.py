# some initializtion sequence, (lights flash different colors)
import threading
import time
import numpy as np
import mediapipe as mp
import cv2
from rover import constants
import signal
import gpiozero
from rover.sonar import Sonar
from rover.sonar_led import SonarLEDS
from rover.motor import Motor
from rover.drivetrain import Drivetrain
from rover.servo import Servo
from rover import constants
import tensorflow as tf
import os
drivetrain = Drivetrain()
movement = ""
moving = 0
minDist = 1
stopped = 0
os.environ["CPUINFO_LOG_LEVEL"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
inverted = False


# Load the TFLite model and allocate tensors
model_path = 'hand_gesture_model.tflite'

# Create GPU delegate options
# try:
#     # Try to use the GPU delegate
#     delegate_options = tf.lite.experimental.delegates.GPU.DelegateOptions()
#     gpu_delegate = tf.lite.experimental.delegates.GPU.Delegate(delegate_options)
    
#     # Initialize interpreter with GPU delegate
#     interpreter = tf.lite.Interpreter(
#         model_path=model_path,
#         experimental_delegates=[gpu_delegate]
#     )
#     print("Using TensorFlow Lite GPU delegate")
# except Exception as e:
#     # Fall back to CPU if GPU delegate fails
#     print(f"GPU acceleration unavailable: {e}")
#     print("Falling back to CPU")

interpreter = tf.lite.Interpreter(model_path='hand_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Gesture mapping
gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]


counter = 0
curr_gest = ""
buzzer = gpiozero.Buzzer(constants.BUZZER_PIN)
sonar = Sonar()
sonar_leds = SonarLEDS()
# Function to normalize landmarks
#def signal_handler(sig, frame):
#        buzzer.off()
#        exit(0)
#signal.signal(signal.SIGINT, signal_handler)
def normalize_landmarks(landmarks):
    # Take the first landmark as the reference point (0, 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()
pan_servo = Servo(constants.CAMERA_SERVOS['pan'])
tilt_servo = Servo(constants.CAMERA_SERVOS['tilt'])
pan_servo.set_angle(90) # default initialization
time.sleep(1)
tilt_servo.set_angle(140) # default initialization
time.sleep(1)
pan_angle = 90.0
tilt_angle = 140.0

def rotation(targetangle):
    global tilt_angle
    global inverted
    print(targetangle)
    if (targetangle >= 0.0 and targetangle <= 180.0) and inverted == False:
        pan_servo.set_angle(int(targetangle))
    elif (targetangle >= 0.0 and targetangle <= 180.0) and inverted == True:
        inverted = False
        tilt_angle = 180.0 - tilt_angle
        tilt_servo.set_angle(int(tilt_angle))
        time.sleep(0.15)
        pan_servo.set_angle(int(targetangle))
    elif inverted == False:
        inverted = True
        tilt_angle = 180.0 - tilt_angle
        tilt_servo.set_angle(int(tilt_angle))
        time.sleep(0.15)
        pan_servo.set_angle(int(targetangle - 180.0))
    else:
        pan_servo.set_angle(int(targetangle - 180.0))
    time.sleep(0.2)

def tilting(tilt_angle):
    tilt_servo.set_angle(int(tilt_angle))

def update_servos(human_x, human_y):
    global pan_angle
    global tilt_angle
    if human_x <= 0.40 or human_x >= 0.6: # if adequately centered, do nothign 
        pan_angle = pan_angle + 10.0 * (0.5 - human_x)
        if pan_angle < 0:
           pan_angle = 0
        if pan_angle > 180:
           pan_angle = 180
        rotation(pan_angle)
    if human_y <= 0.4 or human_y >= 0.6: 
        tilt_angle = tilt_angle - 5.0 * (0.5 - human_y)
        if tilt_angle > 180:
            tilt_angle = 180
        if tilt_angle < 90:
            tilt_angle = 90
        tilting(tilt_angle)

# def followhuman():
#     global moving
#     global stopped 
#     global curr_gest
#     global cap
#     global inverted
#     global wrist
#     global ret
#     global frame
#     global result
#     global wristx
#     global wristy
#     # Get video properties
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Flip the frame horizontally for a mirror effect
#         # frame = cv2.flip(frame, 1)
#         # frame = cv2.resize(frame, (width, height))
#         # # Convert BGR image to RGB
#         # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # # cv2.imshow('Hand Gesture Recognition', frame)
#         # # Process the frame and get hand landmarks
#         # result = hands.process(rgb_frame)

#         if result.multi_hand_landmarks:
#             # for landmarks in result.multi_hand_landmarks:
#             #     mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

#                 # # Get the wrist position
#                 # wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
#                 # print(f"Wrist Position: (x: {wrist.x:.2f}, y: {wrist.y:.2f})")
#                 # if inverted:
#                 #     print("inverted")
#                 #     update_servos(1 - wrist.x, 0.5) # FIXME
#                 # else:
#                 #     print("not inverted")
#                 #     update_servos(wrist.x, 0.5) # FIXME
#             pass
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     time.sleep(2)
def lights():
    global moving
    global stopped
    global curr_gest
    global buzzer
    global sonar
    global sonar_leds
    # while True:
    if (not moving):
        sonar_leds.left.setPixelColor(0x00FF00)
        sonar_leds.right.setPixelColor(0x00FF00)
    else:
        sonar_leds.left.setPixelColor(0x0000FF)
        sonar_leds.right.setPixelColor(0x0000FF)
    dist = sonar.get_distance()
    # while(dist < 300):
    #     stopped = 1
    #     print("im stuck")
    #     sonar_leds.left.setPixelColor(0xFF0000)
    #     sonar_leds.right.setPixelColor(0xFF0000)
    #   # buzzer.on()
    #     counter = 0
    #     while(counter < 3):
    #         dist = sonar.get_distance()
    #         time.sleep(.5)
    #         counter+=1
    #         if(dist > 300): break
    #     counter = 0
    #     dist = sonar.get_distance()
    #     while counter < 3:
    #         dist = sonar.get_distance()
    #         counter +=1
    #         time.sleep(0.5)
    #         if(dist > 300): break
    #    # buzzer.off()
        # dist = sonar.get_distance()
        
        # time.sleep(0.5)
        # if(dist > 300): break
    # simple code
    if dist<300:
        print("im stuck")
        stopped = 1
        sonar_leds.left.setPixelColor(0xFF0000)
        sonar_leds.right.setPixelColor(0xFF0000)
        time.sleep(0.5)
        # continue
        return 0
    stopped = 0
    return 1

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0
):
    return (
        f"v4l2src device=/dev/video{sensor_id} ! "
        f"video/x-raw, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"videoconvert ! videoscale ! "
        f"video/x-raw, width={display_width}, height={display_height} ! "
        "appsink max-buffers=1 drop=True"
    )
# Configure and start the camera with GStreamer
print("Starting video capture with GStreamer...")

def motors():
    global moving
    global stopped
    global curr_gest
    global movement
    global width
    global height
    cap = cv2.VideoCapture(0)


    # cap = cv2.VideoCapture(0)
   #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = 640
    height = 480
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.65)
    # Initialize drawing utils for landmarks
    mp_drawing = mp.solutions.drawing_utils
    while True:
        wristx=0
        wristy=0
            #print("ima capper")
        ret, frame = cap.read()
        if not ret:
            print("not ret -> broken")
            break
        frame = cv2.resize(frame, (width, height))
        frame = cv2.flip(frame,0)
        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to find hands
        result = hands.process(rgb_frame)
        notfound = 0
        if result.multi_hand_landmarks:
            print("ima hand")
            for hand_landmarks in result.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                print(gesture_name)
                if gesture_name == curr_gest:
                    counter+=1
                else:
                    counter = 0
                    curr_gest = gesture_name
                if counter >= 1:
                        movement = gesture_name
               # Get the wrist position
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                print(f"Wrist Position: (x: {wrist.x:.2f}, y: {wrist.y:.2f})")
                wristx = wrist.x
                wristy = wrist.y
                if inverted:
                    print("inverted")
                    update_servos(wrist.x, wrist.y) # FIXME
                else:
                    print("not inverted")
                    update_servos(1 - wrist.x, wrist.y) # FIXME
        else:
            print("No hand detected")
            wristx = 10000
            wristy = 10000  
            drivetrain.front_left_motor.stop()
            drivetrain.front_right_motor.stop()
            drivetrain.rear_left_motor.stop()
            drivetrain.rear_right_motor.stop()
            notfound = 1
        cv2.imshow('Hand Gesture Recognition', frame)

        # if notfound==1:           
        #     continue
        # # --------------
        # go = lights()
        # if go==0:
        #     continue
        # --------------
        # if(stopped):
        #     drivetrain.set_motion(speed=50, heading=270)
        #     time.sleep(0.1)
        #     if(not stopped):
        #         drivetrain.stop()
        #     continue
        # if((wristx-0.5)**2+(wristy-0.5)**2 > minDist**2):
        #     print("fucked")
        #     moving = 0
        #     continue
            #print("Wrist Distance",np.sqrt((wristx-0.5)**2+(wristy-0.5)**2))
        # cap.grab()
        if movement == 'Right':
            print(movement)
            moving = 1
            if(not inverted): drivetrain.set_motion(speed=40, heading=180)
            else: drivetrain.set_motion(speed=40, heading=0)
        elif movement == 'Left':
            print(movement)
            moving = 1
            if(not inverted): drivetrain.set_motion(speed=40, heading=0)
            else: drivetrain.set_motion(speed=40, heading=180)
        elif movement == 'Down':
            moving = 1
            print(movement)
            if(not inverted): drivetrain.set_motion(speed=40, heading=90)
            else: drivetrain.set_motion(speed=40, heading=270)
        elif movement == 'Up' or movement == 'Fire':
            print('Up')
            moving = 1
            if(not inverted): drivetrain.set_motion(speed=40, heading=270)
            else: drivetrain.set_motion(speed=40, heading=90)
        else:
            drivetrain.set_motion(speed=0, heading=180)
            drivetrain.front_left_motor.stop()
            drivetrain.front_right_motor.stop()
            drivetrain.rear_left_motor.stop()
            drivetrain.rear_right_motor.stop()
            moving = 0
            movement = ""
        # movement = ""
        #center camera
        #take photo
        #get command (neutral, go closer, go further, stop, rotate, etc...)
        #if command == go closer:
        #    do stuff'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    
motors()
# print("here")
    

