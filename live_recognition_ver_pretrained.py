import os
import torch
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
import cv2
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights

# Define the face recognition model


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load the pre-trained ResNet50 model
        # weights=ResNet50_Weights.IMAGENET1K_V2
        self.base_model = resnet50(weights=None)
        # Replace the last fully connected layer (classifier) for face recognition
        self.base_model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Perform the forward pass of the model
        x = self.base_model(x)
        return x


# Specify the root directory of the dataset
root_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos'

# Define the number of classes based on the number of folders in the dataset
num_classes = len(os.listdir(root_dir))

# Create an instance of the face recognition model
model = FaceRecognitionModel(num_classes)

# Specify the path to the saved model's state dictionary
model_path = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\face_recognition_model_V2weights_pretrained.pt'

# Load the saved model's state dictionary
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Define a dictionary to map class indices to class labels
class_labels = {idx: folder_name for idx,
                folder_name in enumerate(os.listdir(root_dir))}

# Define a label for unrecognized faces
unknown_label = "Unknown"

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Define the face detection classifier (e.g., Haar cascade or deep learning-based face detector)
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\haarcascade_frontalface_default.xml')

# Set the device for inference (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

while True:
    # Capture frame-by-frame from the video feed
    ret, frame = video_capture.read()

    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=4, minSize=(224, 224))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]

        # Convert the preprocessed face image to a PyTorch tensor
        face_tensor = transforms.ToTensor()(face)

        # Add an extra dimension (batch dimension)
        face_tensor = face_tensor.unsqueeze(0)

        # Move the face tensor to the device
        face_tensor = face_tensor.to(device)
        # Set the confidence threshold
        confidence_threshold = 0.85

        # Perform face recognition on the face tensor
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Forward pass
            outputs = model(face_tensor)

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
