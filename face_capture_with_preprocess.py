import cv2
import os

# Set up the camera and face detection cascade classifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prompt the user to enter the number of users
num_users = int(input("Enter the number of users: "))

# Create directories for each user to store the images
for user_id in range(1, num_users + 1):
    user_name = input("Enter the name for User {}: ".format(user_id))
    user_dir = os.path.join(r"C:\Users\User\Desktop\deep_learn_project\deep_learn_project\photos", user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Capture several images of the user's face
    count = 0
    while count < 200:
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100))
        for (x, y, w, h) in faces:
            # Crop the image to the face region
            face_img = frame[y:y+h, x:x+w]

            # Preprocess the captured face image
            resized_face = cv2.resize(face_img, (224, 224))
            normalized_face = resized_face / 255.0
            normalized_face *= 255.0
            normalized_face = normalized_face.astype("uint8")

            # Save the preprocessed face image
            cv2.imwrite("{}/{}_{}_preprocessed.jpg".format(user_dir, user_name, count), normalized_face)

            count += 1
            cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()


