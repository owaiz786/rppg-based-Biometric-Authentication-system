import cv2
import sys
import os

def test_face_detection(image_path):
    """Test if a face can be detected in an image"""
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
    
    print(f"Image shape: {img.shape}")
    
    # Try OpenCV face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    print(f"OpenCV detected {len(faces)} faces")
    
    if len(faces) > 0:
        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the result
        output_path = "detected_faces.jpg"
        cv2.imwrite(output_path, img)
        print(f"Saved face detection result to {output_path}")
        return True
    else:
        print("No faces detected")
        return False

if __name__ == "__main__":
    # You can pass an image path as argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            test_face_detection(image_path)
        else:
            print(f"File not found: {image_path}")
    else:
        print("Please provide an image path")
        print("Usage: python test_face_detection.py <image_path>")