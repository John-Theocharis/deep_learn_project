import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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
batch_size = 120

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
# optimizer = optim.Adam(model.parameters(), lr=0.001,
#                        weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Set the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define the number of training epochs
num_epochs = 5

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Initialize lists to store training and validation F1 scores
train_f1_scores = []
val_f1_scores = []

# Initialize list to trore training and validatoin accuracies
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Initialize variables for loss and f1 score calculation
    train_loss = 0.0
    train_f1 = 0.0
    total_train_samples = 0
    total_train_correct = 0

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
        train_loss += loss.item() * images.size(0)

        # Backward pass
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        # Calculate the F1 score
        _, predicted = torch.max(outputs, 1)
        train_f1 += f1_score(labels.cpu(), predicted.cpu(), average='macro')
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()

    # Calculate the average training loss, F1 score, and accuracy for the epoch
    train_loss /= len(train_dataset)
    train_f1 /= total_train_samples
    train_accuracy = total_train_correct / total_train_samples

    # Append the training loss, F1 score, and accuracy to the respective lists
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    train_accuracies.append(train_accuracy)

    # Print the loss, F1 score, and accuracy for this epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train F1 Score: {train_f1:.4f}, Train Accuracy: {train_accuracy:.2%}")

    # Validation
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        val_loss = 0.0
        val_f1 = 0.0
        total_val_samples = 0
        total_val_correct = 0

        # Iterate over the validation dataset
        for images, labels in val_loader:
            # Move the data to the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # Calculate the F1 score
            _, predicted = torch.max(outputs, 1)
            val_f1 += f1_score(labels.cpu(), predicted.cpu(), average='macro')
            total_val_samples += labels.size(0)
            total_val_correct += (predicted == labels).sum().item()

        # Calculate the average validation loss, F1 score, and accuracy for the epoch
        val_loss /= len(val_dataset)
        val_f1 /= total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        # Append the validation loss, F1 score, and accuracy to the respective lists
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        val_accuracies.append(val_accuracy)
        # Print the validation loss, F1 score, and accuracy for this epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.2%}")


# Specify the path to save the trained model
model_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project'
model_filename = "face_recognition_model.pt"
model_path = os.path.join(model_dir, model_filename)

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)


# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# Plot the training and validation losses
ax1.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
ax1.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Training and Validation Losses')

# Plot the training and validation F1 scores
ax2.plot(range(1, num_epochs+1), train_f1_scores, label='Training F1 Score')
ax2.plot(range(1, num_epochs+1), val_f1_scores, label='Validation F1 Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.legend()
ax2.set_title('Training and Validation F1 Scores')

# Plot the training and validation accuracies
ax3.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
ax3.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.legend()
ax3.set_title('Training and Validation Accuracies')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Save the plot as an image file
plt.savefig('training_metrics.png')

# Display the plot
plt.show()
