import os
import cv2
from face_detector import FaceDetector


def process_images(source_dir, target_dir):
    detector = FaceDetector()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)
            faces = detector.detect_faces(img)
            for (x, y, w, h) in faces:
                face_crop = img[y:y + h, x:x + w]
                face_crop = cv2.resize(face_crop, (250, 250))

                # Creating same directory structure under target_dir
                rel_path = os.path.relpath(subdir, source_dir)
                new_subdir = os.path.join(target_dir, rel_path)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)

                # Saving cropped face in new directory
                new_img_path = os.path.join(new_subdir, file)
                cv2.imwrite(new_img_path, face_crop)


if __name__ == "__main__":
    process_images('data/dataset', 'data/cropped')
