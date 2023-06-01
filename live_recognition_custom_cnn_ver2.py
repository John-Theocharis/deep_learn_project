import os
import cv2
import torch.nn as nn
from torchvision import transforms
import torch


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.base_model = CustomModel(num_classes)

    def forward(self, x):
        features = self.base_model(x)
        return features


# Specify the path to the saved model's state dictionary
model_path = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\face_recognition_custom_model_dict.pt'

# Specify the root directory of the dataset
root_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos'

# Define the number of classes based on the number of folders in the dataset
num_classes = len(os.listdir(root_dir))

# Create an instance of the face recognition model
model = FaceRecognitionModel(num_classes=num_classes)

# Load the saved model's state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Specify the root directory of the dataset
root_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos'

# Define a dictionary to map class indices to class labels
class_labels = {idx: folder_name for idx,
                folder_name in enumerate(os.listdir(root_dir))}

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the face detection classifier (e.g., Haar cascade or deep learning-based face detector)
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\haarcascade_frontalface_default.xml')


def preprocess_face(image):
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(224, 224))

    # Preprocess each detected face
    preprocessed_faces = []
    for (x, y, w, h) in faces:
        # Crop the face region
        face = image[y:y+h, x:x+w]

        # Resize the face to a fixed size
        resized_face = cv2.resize(face, (224, 224))

        # Normalize the pixel values to [0, 1]
        normalized_face = resized_face / 255.0

        # Convert the normalized face back to the range [0, 255]
        normalized_face *= 255.0
        normalized_face = normalized_face.astype("uint8")

        preprocessed_faces.append(normalized_face)

    return preprocessed_faces


while True:
    # Capture frame-by-frame from the video feed
    ret, frame = video_capture.read()

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=4, minSize=(224, 224))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face image
        preprocessed_faces = preprocess_face(face)

        # Iterate over preprocessed faces
        for preprocessed_face in preprocessed_faces:
            # Convert the preprocessed face to tensor
            tensor_face = transforms.ToTensor()(preprocessed_face)
            tensor_face = tensor_face.unsqueeze(0)

            # Set the device for inference (CPU or GPU)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

            # Move the preprocessed face tensor to the device
            tensor_face = tensor_face.to(device)

            # Perform face recognition on the preprocessed face tensor
            with torch.no_grad():
                # Forward pass
                outputs = model(tensor_face)

                # Get the predicted class label and confidence scores
                _, predicted_idx = torch.max(outputs, 1)
                predicted_label = class_labels[predicted_idx.item()]
                confidence_score = torch.softmax(outputs, dim=1)[
                    0, predicted_idx].item()
                # Set the confidence threshold
                confidence_threshold = 0.85
                # Check if the confidence score is above the threshold
                if confidence_score >= confidence_threshold:
                    # Draw bounding box and label on the frame for recognized face
                    # Include confidence score in label
                    label_text = f"{predicted_label}: {confidence_score:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    # Draw bounding box and label on the frame for unknown face
                    # Include confidence score in label
                    label_text = f"Unknown: {confidence_score:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Live Video', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
