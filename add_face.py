import cv2
import os
from utils.face_detector import FaceDetector


def add_face(user_name):
    face_detector = FaceDetector()

    cap = cv2.VideoCapture(0)

    # Create a new directory for the user
    os.makedirs(f'data/train/{user_name}', exist_ok=True)

    while True:
        ret, frame = cap.read()

        faces = face_detector.detect_faces(frame)

        # Save detected faces
        for i, (x, y, w, h) in enumerate(faces):
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(f'data/train/{user_name}/{i}.jpg', face_img)

        cv2.imshow('frame', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys

    username = sys.argv[1]  # Get the username from the command line argument
    add_face(username)
