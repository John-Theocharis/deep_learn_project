import cv2
import os

# Set the path to the directory containing the random face images
random_faces_dir = r"C:\Users\User\Desktop\deep_learn_project\photos for training\original\Haris"

# Create a new directory to store the preprocessed random face images
preprocessed_dir = r"C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos\HarisA"
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocess the random face images
for filename in os.listdir(random_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image_path = os.path.join(random_faces_dir, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=1, minSize=(224, 224))

        # Crop and preprocess each detected face
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

            # Save the preprocessed random face image
            output_filename = os.path.splitext(
                filename)[0] + "_preprocessed.jpg"
            output_path = os.path.join(preprocessed_dir, output_filename)
            cv2.imwrite(output_path, normalized_face)


# Close the windows
cv2.destroyAllWindows()
