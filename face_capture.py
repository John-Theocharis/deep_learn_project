import cv2
import os

# Set up the camera and face detection cascade classifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prompt the user to enter the number of users
num_users = int(input("Enter the number of users: "))

# Create directories for each user to store the images
for user_id in range(1, num_users + 1):
    user_name = input("Enter the name for User {}: ".format(user_id))
    user_dir = os.path.join(
        r"C:\Users\John\Desktop\dimokritos\deep_learn_folder\deep_learn_project\photos", user_name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    # Capture several images of the user's face
    count = 0
    while count < 600:
        ret, frame = cap.read()
        faces = face_cascade.detectMultiScale(
            frame, scaleFactor=1.2, minNeighbors=7, minSize=(100, 100))
        for (x, y, w, h) in faces:
            # Crop the image to the face region
            # face_img = frame[y:y+h, x:x+w]
            # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # # Save the cropped face image
            # cv2.imwrite("{}/{}_{}_color.jpg".format(user_dir, user_name, count), face_img)

            cv2.imwrite("{}/{}_{}_color.jpg".format(user_dir,
                        user_name, count), frame)

            count += 1
            cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
