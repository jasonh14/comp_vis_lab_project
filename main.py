import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split


dataset_path = "./Dataset"
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
person_names = os.listdir(dataset_path)

def load_dataset():
    data = []
    labels = []
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_path):
            for img_filename in os.listdir(class_path):
                img_path = os.path.join(class_path, img_filename)
                # img = cv.imread(img_path, 0)
               
                data.append(img_path)
                labels.append(class_name)
    
    return data, labels

def train(X_train):
    face_list = []
    class_list = []
    for idx, img_path in enumerate(X_train):
        img = cv.imread(img_path, 0)
        detected_face = haarcascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)

        if len(detected_face) < 1:
            continue

        for face_coor in detected_face:
            x,y,h,w = face_coor
            face_img = img[y:y+h, x:x+w]
            face_list.append(face_img)
            class_list.append(idx)

    return face_list, class_list

def test(X_test, face_recognizer):
    for img_path in X_test:
        img_gray = cv.imread(img_path, 0)
        img_bgr = cv.imread(img_path)
        detected_face = haarcascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        if len(detected_face) < 1:
            continue

        for face_coor in detected_face:
            x,y,h,w = face_coor
            face_img = img_gray[y:y+h, x:x+w]
            res, confidence = face_recognizer.predict(face_img)
            cv.rectangle(img_bgr, (x, y), (x+w, y+h), (255, 0, 0), 1)
            print("res", res)
            # text = str(confidence)
            cv.putText(img_bgr, str(person_names[res]) + " : " + str(confidence) , (x, y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv.imshow("Result", img_bgr)
            cv.waitKey(0)
            

def main():
   
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    face_list, class_list = train(X_train)
    class_names = list(set(class_list))
    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    face_recognizer.train(face_list, np.array(class_list))
    test(X_test, face_recognizer)

    
    print("Done")

   
main()










