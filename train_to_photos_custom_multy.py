import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, multilabel_confusion_matrix


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

        #  inspect tensor size
        # print(x.size())

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


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = self.get_image_paths()
        self.classes = self.get_classes()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                      0.229, 0.224, 0.225])
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


# Set the random seed for reproducibility
torch.manual_seed(0)

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
batch_size = 32

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
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Set the device for training (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Define the number of training epochs
num_epochs = 15

# Import the necessary libraries

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Initialize lists to store training and validation F1 scores
train_f1_scores = []
val_f1_scores = []

# Initialize list to store training and validation accuracies
train_accuracies = []
val_accuracies = []

# Initialize lists to store true positives, true negatives, false positives, and false negatives
train_tp = []
train_tn = []
train_fp = []
train_fn = []

val_tp = []
val_tn = []
val_fp = []
val_fn = []

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

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

        # Calculate the true positives, true negatives, false positives, and false negatives
        _, predicted = torch.max(outputs, 1)
        tp = ((predicted == labels) & (predicted == 1)).sum().item()
        tn = ((predicted == labels) & (predicted == 0)).sum().item()
        fp = ((predicted != labels) & (predicted == 1)).sum().item()
        fn = ((predicted != labels) & (predicted == 0)).sum().item()

        train_tp.append(tp)
        train_tn.append(tn)
        train_fp.append(fp)
        train_fn.append(fn)

        # Append the true labels and predicted labels for the training set
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # Calculate the F1 score
        train_f1 += f1_score(labels.cpu().numpy(),
                             predicted.cpu().numpy(), average='macro')

        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()

    # Calculate the average training loss, F1 score, and accuracy for the epoch
    train_loss /= len(train_dataset)
    train_f1 /= len(train_loader)
    train_accuracy = total_train_correct / total_train_samples

    # Append the training loss, F1 score, and accuracy to the respective lists
    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    train_accuracies.append(train_accuracy)

    # Print the loss, F1 score, and accuracy for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train F1 Score: {train_f1:.4f}, Train Accuracy: {train_accuracy:.2%}")

    # Validation
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation
    with torch.no_grad():
        val_loss = 0.0
        val_f1 = 0
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

            # Calculate the true positives, true negatives, false positives, and false negatives
            _, predicted_val = torch.max(outputs, 1)
            tp = ((predicted_val == labels) & (
                predicted_val == 1)).sum().item()
            tn = ((predicted_val == labels) & (
                predicted_val == 0)).sum().item()
            fp = ((predicted_val != labels) & (
                predicted_val == 1)).sum().item()
            fn = ((predicted_val != labels) & (
                predicted_val == 0)).sum().item()

            val_tp.append(tp)
            val_tn.append(tn)
            val_fp.append(fp)
            val_fn.append(fn)

            # Append the true labels and predicted labels for the validation set
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted_val.cpu().numpy())

            # Calculate the F1 score
            val_f1 += f1_score(labels.cpu().numpy(),
                               predicted_val.cpu().numpy(), average='macro')

            total_val_samples += labels.size(0)
            total_val_correct += (predicted_val == labels).sum().item()

        # Calculate the average validation loss, F1 score, and accuracy for the epoch
        val_loss /= len(val_dataset)
        val_f1 /= len(val_loader)
        val_accuracy = total_val_correct / total_val_samples

        # Append the validation loss, F1 score, and accuracy to the respective lists
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        val_accuracies.append(val_accuracy)

        # Print the validation loss, F1 score, and accuracy for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.4f}, Validation Accuracy: {val_accuracy:.2%}")

# Convert the predicted labels and true labels to NumPy arrays
predicted_labels = np.array(predicted_labels)
true_labels = np.array(true_labels)

# Calculate the multilabel confusion matrix for training data
train_mcm = multilabel_confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix (Training):")
for i, mcm in enumerate(train_mcm):
    print(f"Class {i}:")
    print(mcm)
    print()

# Calculate the overall accuracy for training data
train_overall_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy (Training): {train_overall_accuracy:.2%}")

# Calculate the multilabel confusion matrix for validation data
val_mcm = multilabel_confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix (Validation):")
for i, mcm in enumerate(val_mcm):
    print(f"Class {i}:")
    print(mcm)
    print()

# Calculate the overall accuracy for validation data
val_overall_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy (Validation): {val_overall_accuracy:.2%}")

# Calculate classification metrics
classification_metrics = classification_report(true_labels, predicted_labels)
print(classification_metrics)


# Specify the path to save the trained model
model_dir_dict = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project'
model_filename = "face_recognition_custom_model_dict.pt"
full_model = "face_recognition_custom_full_model.pt"

model_path_dict = os.path.join(model_dir_dict, model_filename)
whole_model_path = os.path.join(model_dir_dict, full_model)
# Save the model's state dictionary
torch.save(model.state_dict(), model_path_dict)

# Save the whole model
torch.save(model, whole_model_path)

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
plt.savefig('training_metrics_custom_model.png')

# Display the plot
plt.show()
