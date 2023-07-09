import cv2
import torch

from util.logger import logger
from util.prepare_data import define_model


def run_model(args):
    selected_model, num_classes, class_to_label, device, face_cascade = __init_data(args)

    # Load the model
    model = define_model(selected_model, num_classes).to(device)
    model.load_state_dict(torch.load(f'models/pth/{selected_model}.pth', map_location=device))

    # Start the camera feed
    logger.info('Starting camera feed...')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Get the face
            face = frame[y:y + h, x:x + w]

            # Preprocess the face
            face = cv2.resize(face, (250, 250))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face / 255.0
            face = torch.from_numpy(face).float()
            face = face.permute(2, 0, 1)
            face = face.unsqueeze(0).to(device)

            # Predict the class of the face
            output = model(face)
            _, predicted = torch.max(output, 1)

            # Convert the predicted class back into a name
            predicted_name = class_to_label[str(predicted.item())]

            # Draw the name onto the frame
            cv2.putText(frame, predicted_name, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info('Stopping camera feed...')
            break

    cap.release()
    cv2.destroyAllWindows()


def __init_data(args):
    return args.selected_model, \
        args.num_classes, \
        args.class_to_label, \
        args.device, \
        args.face_cascade
