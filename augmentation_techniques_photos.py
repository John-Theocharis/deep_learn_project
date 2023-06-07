import cv2
import os
import torch
from torchvision import transforms

# Set the path to the directory containing the random face images
random_faces_dir = r"C:\Users\User\Desktop\deep_learn_project\photos for training\original\Haris"

# Create a new directory to store the augmented and preprocessed random face images
augmented_dir = r"C:\Users\User\Desktop\deep_learn_project\photos for training\original\Haris_augment"
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define the transformations for augmentation
augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.2),
    transforms.ToTensor()
])

# Preprocess and augment the random face images
for filename in os.listdir(random_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(random_faces_dir, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(224, 224))

        # Preprocess and augment each detected face
        for (x, y, w, h) in faces:
            # Check if the face size is larger than the crop size
            if h >= 224 and w >= 224:
                # Crop the face region
                face = image[y:y+h, x:x+w]

                # Resize the face to a fixed size
                resized_face = cv2.resize(face, (224, 224))

                # Normalize the pixel values to [0, 1]
                normalized_face = resized_face / 255.0

                # Rearrange dimensions: (H, W, C) to (C, H, W)
                normalized_face = normalized_face.transpose((2, 0, 1))

                # Convert the normalized face to a PyTorch tensor
                tensor_face = torch.from_numpy(normalized_face)

                # Apply augmentations to the face
                augmented_face = augmentation_transforms(tensor_face)

                # Convert the augmented face to a NumPy array
                augmented_face_np = augmented_face.numpy()

                # Rearrange dimensions: (C, H, W) to (H, W, C)
                augmented_face_np = augmented_face_np.transpose((1, 2, 0))

                # Convert the augmented face back to the range [0, 255]
                augmented_face_np *= 255.0
                augmented_face_np = augmented_face_np.astype("uint8")

                # Save the augmented and preprocessed random face image
                output_filename = os.path.splitext(
                    filename)[0] + "_augmented.jpg"
                output_path = os.path.join(augmented_dir, output_filename)
                cv2.imwrite(output_path, augmented_face_np)

# Close the windows
cv2.destroyAllWindows()
