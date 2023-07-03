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
        # weights=ResNet50_Weights.IMAGENET1K_V2
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.base_model.fc = nn.Linear(2048, num_classes)

        # freeze the first three layers
        for name, module in self.base_model.named_children():
            if name in ['layer1', 'layer2', 'layer3']:
                for param in module.parameters():
                    param.requires_grad = False

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


root_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos'
dataset = FaceDataset(root_dir)

validation_percentage = 0.2
num_validation_samples = int(validation_percentage * len(dataset))

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - num_validation_samples, num_validation_samples])

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


model = FaceRecognitionModel(num_classes=len(dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 6

train_losses = []
train_f1_scores = []
train_accuracies = []
val_losses = []
val_f1_scores = []
val_accuracies = []

for epoch in range(num_epochs):
    train_loss = 0.0
    train_f1 = 0.0
    total_train_samples = 0
    total_train_correct = 0

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        train_f1 += f1_score(labels.cpu(), predicted.cpu(), average='macro')
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_dataset)
    train_f1 /= len(train_loader)  # Calculate average F1 score per epoch
    train_accuracy = total_train_correct / total_train_samples

    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    train_accuracies.append(train_accuracy)

    val_loss = 0.0
    val_f1 = 0.0
    total_val_samples = 0
    total_val_correct = 0

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            val_f1 += f1_score(labels.cpu(), predicted.cpu(), average='macro')
            total_val_samples += labels.size(0)
            total_val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_dataset)
    val_f1 /= len(val_loader)  # Calculate average F1 score per epoch
    val_accuracy = total_val_correct / total_val_samples

    val_losses.append(val_loss)
    val_f1_scores.append(val_f1)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train F1 Score: {train_f1:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val F1 Score: {val_f1:.4f}, Val Accuracy: {val_accuracy:.4f}")

model_dir = r'C:\Users\User\Desktop\deep_learn_project\deep_learn_project'
model_filename = "face_recognition_model_V2weights_pretrained.pt"
model_path = os.path.join(model_dir, model_filename)

torch.save(model.state_dict(), model_path)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

ax1.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
ax1.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Training and Validation Losses')

ax2.plot(range(1, num_epochs+1), train_f1_scores, label='Training F1 Score')
ax2.plot(range(1, num_epochs+1), val_f1_scores, label='Validation F1 Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.legend()
ax2.set_title('Training and Validation F1 Scores')

ax3.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
ax3.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.legend()
ax3.set_title('Training and Validation Accuracies')

plt.subplots_adjust(hspace=0.4)

plt.savefig('training_metrics_V2weights_pretrained.png')

plt.show()
