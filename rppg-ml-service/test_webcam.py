import cv2
import time

def test_webcam():
    """Test webcam capture and face detection with better error handling"""
    
    print("Testing webcam access...")
    
    # Try different backends on Windows
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),  # Preferred on Windows
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Any")
    ]
    
    for camera_id in [0, 1]:
        for backend, backend_name in backends:
            print(f"\nTrying camera {camera_id} with {backend_name} backend...")
            
            # Try to open with specific backend
            cap = cv2.VideoCapture(camera_id, backend)
            
            if not cap.isOpened():
                print(f"Could not open camera {camera_id} with {backend_name}")
                continue
            
            # Set properties for better compatibility
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Give camera time to initialize
            time.sleep(1)
            
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret and frame is not None:
                print(f"SUCCESS! Captured frame with shape: {frame.shape}")
                print(f"Frame stats - Mean: {frame.mean():.2f}, Min: {frame.min()}, Max: {frame.max()}")
                
                # Save the frame
                filename = f"webcam_test_cam{camera_id}_{backend_name}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame to {filename}")
                
                # Try face detection
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                print(f"Face detection: {len(faces)} faces found")
                
                if len(faces) > 0:
                    # Draw rectangles
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    cv2.imwrite(f"webcam_test_cam{camera_id}_{backend_name}_with_faces.jpg", frame)
                    print(f"Saved face detection result")
                
                cap.release()
                return True
            else:
                print(f"Could not read frame from camera {camera_id} with {backend_name}")
                cap.release()
    
    print("\nNo working camera found!")
    return False

def list_cameras():
    """List available cameras"""
    print("\nScanning for available cameras...")
    
    # Try different APIs
    for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        api_name = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation", cv2.CAP_ANY: "Any"}.get(api, "Unknown")
        print(f"\nChecking with {api_name} API:")
        
        for i in range(5):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                print(f"  Camera {i}: Available with {api_name}")
                # Try to get camera name (Windows specific)
                try:
                    # Set a small frame size for testing
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        print(f"    Can capture frames: Yes ({frame.shape})")
                    else:
                        print(f"    Can capture frames: No")
                except:
                    print(f"    Can capture frames: Error")
                cap.release()
            else:
                print(f"  Camera {i}: Not available with {api_name}")

if __name__ == "__main__":
    print("=" * 50)
    print("WEBCAM TEST SCRIPT")
    print("=" * 50)
    
    # List available cameras
    list_cameras()
    
    # Test webcam
    print("\n" + "=" * 50)
    print("TESTING WEBCAM CAPTURE")
    print("=" * 50)
    test_webcam()