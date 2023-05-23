import cv2
import os

# Set up the camera and face detection cascade classifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prompt the user to enter their name
name = input("Enter your name: ")

# Create a directory with the user's name to store the images
if not os.path.exists(os.path.join("photos", name)):
    os.makedirs(os.path.join("photos", name))

# Capture several images of the user's face
count = 0
while count < 20:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(
            "{}/{}_{}_color.jpg".format(os.path.join("photos", name), name, count), frame)

        count += 1
        cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
