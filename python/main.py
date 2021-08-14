from keras.models import model_from_json
import numpy as np
import cv2
import time


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Sad", "Surprise",
                     "Neutral"]


with open('model/face_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model/face_model.h5")

loaded_model.save("whole_model.h5")
loaded_model.summary()

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    start = time.time_ns() / 1000000
    gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        faces = faceCascade.detectMultiScale(frame)

        if faces is not None:
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y + h, x:x + w]

                roi = cv2.resize(fc, (48, 48))

            emotion = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
            label = EMOTIONS_LIST[np.argmax(emotion)]

            cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Main Frame ",frame)

            end = time.time_ns() / 1000000
            duration = end-start
            print("Time taken for one frame: ", duration, " miliseconds")
        else:
            print("Face not found!")

        if cv2.waitKey(1) == 27:
            break
