import os
import torch
import cv2
import torch.nn as nn
from torchvision import transforms


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.conv3(x)
        x = self.bn3(x)
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

# Define a dictionary to map class indices to class labels
class_labels = {idx: folder_name for idx,
                folder_name in enumerate(os.listdir(root_dir))}

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the face detection classifier (e.g., Haar cascade or deep learning-based face detector)
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame from the video feed
    ret, frame = video_capture.read()

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=4, minSize=(224, 224))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Preprocess the face image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

        ])
        preprocessed_face = transform(face).unsqueeze(0)

        # Set the device for inference (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the preprocessed face tensor to the device
        preprocessed_face = preprocessed_face.to(device)

        # Set the confidence threshold
        confidence_threshold = 0.6

        # Perform face recognition on the preprocessed face tensor
        with torch.no_grad():
            # Forward pass
            outputs = model(preprocessed_face)

            # Get the predicted class label and confidence scores
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = class_labels[predicted_idx.item()]
            confidence_score = torch.softmax(outputs, dim=1)[
                0, predicted_idx].item()

            # Check if the confidence score is above the threshold
            if confidence_score >= confidence_threshold:
                # Draw bounding box and label on the frame for recognized face
                # Include confidence score in label
                label_text = f"{predicted_label}: {confidence_score:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw bounding box and label on the frame for unknown face
                # Include confidence score in label
                label_text = f"Unknown: {confidence_score:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Live Video', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
