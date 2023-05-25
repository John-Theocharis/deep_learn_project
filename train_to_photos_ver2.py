import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Define the face recognition model


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Load the pre-trained ResNet50 model
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze the first three layers
        for name, module in self.base_model.named_children():
            if name in ['layer1', 'layer2', 'layer3']:
                for param in module.parameters():
                    param.requires_grad = False

        # Reinitialize layer4 and fc with random weights
        # nn.init.xavier_uniform_(self.base_model.layer4.weight)
        # nn.init.zeros_(self.base_model.layer4.bias)
        # nn.init.xavier_uniform_(self.base_model.fc.weight)
        # nn.init.zeros_(self.base_model.fc.bias)

        # Replace the last fully connected layer (classifier) for face recognition
        self.base_model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.base_model(x)
        return features


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = self.get_image_paths()
        self.classes = self.get_classes()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def get_image_paths(self):
        image_paths = []
        for user_folder in os.listdir(self.root_dir):
            user_folder_path = os.path.join(self.root_dir, user_folder)
            if os.path.isdir(user_folder_path):
                for image_file in os.listdir(user_folder_path):
                    image_path = os.path.join(user_folder_path, image_file)
                    image_paths.append(image_path)
        return image_paths

    def get_classes(self):
        classes = set()
        for image_path in self.image_paths:
            label = os.path.basename(os.path.dirname(image_path))
            classes.add(label)
        return sorted(list(classes))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = os.path.basename(os.path.dirname(image_path))
        return image, self.classes.index(label)


# Provide the path to the root directory of the dataset
root_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos'

# Create an instance of the FaceDataset
dataset = FaceDataset(root_dir)
# print(len(dataset))

# Define the percentage of the dataset to use for validation
validation_percentage = 0.2

# Calculate the number of samples for validation
num_validation_samples = int(validation_percentage * len(dataset))

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - num_validation_samples, num_validation_samples])

# Specify the batch size for training and validation data loaders
batch_size = 64

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# Define the face recognition model
model = FaceRecognitionModel(num_classes=len(dataset.classes))

# # Access the list of classes
# classes = dataset.classes

# # Print the names of the classes
# print(classes)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Set the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define the number of training epochs
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Iterate over the training dataset
    for images, labels in train_loader:
        # Move the data to the device
        images = images.to(device)
        labels = labels.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update the model's parameters
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        # Iterate over the validation dataset
        for images, labels in val_loader:
            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Calculate the number of correct predictions
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate the accuracy
        accuracy = total_correct / total_samples

        # Print the validation accuracy for this epoch
        print(f"Validation Accuracy: {accuracy:.2f}")


# Specify the path to save the trained model
model_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project'
model_filename = "face_recognition_model.pt"
model_path = os.path.join(model_dir, model_filename)

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)
