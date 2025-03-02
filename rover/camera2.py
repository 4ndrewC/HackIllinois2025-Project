import typing
import time
from picamera2 import Picamera2
import numpy as np
import cv2
import socket
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import socketserver

class Camera:
    cam: Picamera2
    image_array: np.ndarray
    
    def __init__(self):
        self.image_array = np.ndarray(0)
        try:
            self.cam = Picamera2()
            # Configure for video capturing
            config = self.cam.create_video_configuration(main={"size": (640, 480)})
            self.cam.configure(config)
            self.cam.start(show_preview=False)
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.cam = None
    
    def capture(self):
        if self.cam is not None:
            self.image_array = self.cam.capture_array()
            return self.image_array
        return None
    
    def start_mjpeg_server(self, port=8000):
        """Start an MJPEG HTTP server to view the camera stream in a browser"""
        global frame_to_serve
        frame_to_serve = None
        
        class StreamingHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write("""
                    <html>
                    <head>
                    <title>Raspberry Pi Camera Stream</title>
                    </head>
                    <body>
                    <h1>Raspberry Pi Camera Stream</h1>
                    <img src="/stream.mjpg" width="640" height="480" />
                    </body>
                    </html>
                    """.encode())
                elif self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Age', '0')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                    self.end_headers()
                    try:
                        while True:
                            if frame_to_serve is not None:
                                # Encode frame as JPEG
                                _, img_encoded = cv2.imencode('.jpg', frame_to_serve)
                                jpg_data = img_encoded.tobytes()
                                
                                self.wfile.write(b'--FRAME\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(jpg_data))
                                self.end_headers()
                                self.wfile.write(jpg_data)
                                self.wfile.write(b'\r\n')
                            time.sleep(0.05)
                    except Exception as e:
                        print(f"Streaming error: {e}")
                else:
                    self.send_error(404)
                    self.end_headers()
        
        class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
            allow_reuse_address = True
            daemon_threads = True
            
        # Start frame capture thread
        def capture_frames():
            global frame_to_serve
            while True:
                frame = self.capture()
                if frame is not None:
                    # Convert from camera format to RGB if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_to_serve = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_to_serve = frame
                time.sleep(0.1)
        
        # Get the IP address of the Pi
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't have to be reachable
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
            
        print(f"Starting server on http://{ip}:{port}")
        print(f"Open a browser and navigate to http://{ip}:{port} to view the stream")
        
        # Start frame capture in a thread
        Thread(target=capture_frames, daemon=True).start()
        
        # Start the server
        try:
            address = ('', port)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped by user")

# Example usage
if __name__ == "__main__":
    camera = Camera()
    camera.start_mjpeg_server()