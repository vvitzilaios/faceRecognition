import cv2
import os
from utils.face_detector import FaceDetector
import numpy as np


def add_face(user_name):
    face_detector = FaceDetector()

    cap = cv2.VideoCapture(0)

    # Create a new directory for the user
    os.makedirs(f'data/dataset/{user_name}', exist_ok=True)

    while True:
        ret, frame = cap.read()

        faces = face_detector.detect_faces(frame)

        if np.size(faces) > 0:  # if any faces are detected
            # Find the face with the largest area
            face_areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
            largest_area, largest_face = max(face_areas, key=lambda item: item[0])

            # Draw a green rectangle around the detected face
            x, y, w, h = largest_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the detected face when 's' is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                face_img = frame[y:y + h, x:x + w]
                cv2.imwrite(f'data/dataset/{user_name}/{len(os.listdir(f"data/dataset/{user_name}"))}.jpg', face_img)

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
