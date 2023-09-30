import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset Preparation
dataset_path = "Dataset"
list_person_names = os.listdir(dataset_path)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def train_test():

    X = []
    Y  = [] 

    for idx, name in enumerate(list_person_names):
        person_name_path = dataset_path + "/" + name

        for img_name in os.listdir(person_name_path):
            img_name_path = person_name_path   + "/" + img_name
            img = cv.imread(img_name_path, 0)
            img = cv.bilateralFilter(img, 11, 120, 120)
            img = cv.GaussianBlur(img, (11,11), 0)

            detected_face = haarcascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors=10)

            if len(detected_face) != 1: # stored result has only 1 face detected
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
    print("Training and Testing")
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(X_train, Y_train)

    # Testing
    correct = 0

    for idx, (img_X_test, img_Y_test) in enumerate(zip(X_test,Y_test)):
        # print(f'Tesing {idx} / {len(X_test)}')
        Y_pred, _ = face_recognizer.predict(img_X_test)

        if Y_pred == img_Y_test:
            correct += 1

    accuracy = correct/len(X_test)
    print("Training and Testing Finished")
    print("Average Accuracy: ",accuracy)

    face_recognizer.save("lbph_face_recognizer.xml")

def predict():
    path_img_to_be_predicted = input("Input absolute path for image to predict: ")
    img_to_be_predicted_gray = cv.imread(path_img_to_be_predicted, 0)
    img_to_be_predicted_bgr = cv.imread(path_img_to_be_predicted)

    detected_face = haarcascade.detectMultiScale(img_to_be_predicted_gray, scaleFactor = 1.2, minNeighbors=10)

    model_file = "lbph_face_recognizer.xml"

    if not os.path.isfile(model_file):
        print("Model file not found. Do Training First")
        return
        
    face_recognizer = cv.face_LBPHFaceRecognizer.create()
    face_recognizer.read("lbph_face_recognizer.xml")

    if len(detected_face) < 1:
        raise Exception("Face is not detected")
    
    for face_coor in detected_face:
        x,y,h,w = face_coor
        face_img = img_to_be_predicted_gray[y:y+h, x:x+w]
        Y_pred, confidence = face_recognizer.predict(face_img)

        if (confidence < 100):
            text = list_person_names[Y_pred] + " : " + str(confidence)
        else:
            text = "unknown"

        cv.rectangle(img_to_be_predicted_bgr, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv.putText(img_to_be_predicted_bgr, text, (x,y-10), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv.imshow("Result", img_to_be_predicted_bgr)
        cv.waitKey(0)



def main():
    while True:
        print("Footbal Player Face Recognition")
        print("1. Train and Test Model")
        print("2. Predict")
        print("3. Exit")
        menu = int(input(">> "))

        if(menu == 1):
            train_test()
        elif(menu == 2):
            predict()
        elif(menu == 3):
            break
        else:
            print("Choose correct number")
                
        enter = input("Press enter to continue...")
    
    print("Program Terminated")

main()
