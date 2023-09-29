import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset Preparation
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

X = []
Y  = []

dataset_path = "Dataset"
list_person_names = os.listdir(dataset_path)

for idx, name in enumerate(list_person_names):
    person_name_path = dataset_path + "/" + name

    for img_name in os.listdir(person_name_path):
        img_name_path = person_name_path   + "/" + img_name
        img = cv.imread(img_name_path, 0)
        img = cv.bilateralFilter(img, 5, 200, 200)
        img = cv.GaussianBlur(img, (11,11), 0, None)

        detected_face = haarcascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors=5)


        if len(detected_face) < 1:
            continue
            
        for face_coor in detected_face:
            x,y,h,w = face_coor

            
            face_img = img[y:y+h, x:x+w]
            
            X.append(face_img)
            Y.append(idx)

X = np.array(X, dtype='object')
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

# Training
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(X_train, Y_train)
print("Training face recognition...")

# Testing
correct = 0

for idx, (img_X_test, img_Y_test) in enumerate(zip(X_test,Y_test)):
    print(f'Tesing {idx} / {len(X_test)}')
    Y_pred, _ = face_recognizer.predict(img_X_test)

    if Y_pred == img_Y_test:
        correct += 1

accuracy = correct/len(X_test)
print("accuracy: ",accuracy)

face_recognizer.save("lbph_face_recognizer.xml")




