import typing
from picamera2 import Picamera2
import numpy as np


class Camera:
    cam: Picamera2
    image_array: np.ndarray

    def __init__(self):

        self.image_array = np.ndarray(0)

        try:
            self.cam = Picamera2()
            self.cam.start(show_preview=False)
        except:
            self.cam = None

    def capture(self):

        if self.cam != None:
            self.image_array = self.cam.capture_array()

    # def start_video_stream(self, display=True):
    #         """
    #         Start a continuous video stream
    #         If display is True, shows frames with OpenCV
    #         Returns frames if display is False
    #         """
    #         if self.cam is None:
    #             print("Camera not available")
    #             return
            
    #         try:
    #             while True:
    #                 frame = self.capture()
                    
    #                 if display and frame is not None:
    #                     cv2.imshow("Camera Feed", frame)
    #                     if cv2.waitKey(1) & 0xFF == ord('q'):
    #                         break
    #                 elif not display:
    #                     yield frame
                    
    #                 # Small sleep to prevent maxing out CPU
    #                 time.sleep(0.01)
                    
    #         except KeyboardInterrupt:
    #             print("Stream stopped by user")
    #         finally:
    #             if display:
    #                 cv2.destroyAllWindows()