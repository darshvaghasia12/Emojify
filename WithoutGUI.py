import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageTk
import os
from flask import Flask,render_template,Response
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator



model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))

model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ",
                5: "    Sad    ", 6: "Surprised"}

emoji_dist = {0: "emojis/angry.PNG", 1: "emojis/disgusted.PNG", 2: "emojis/fearful.PNG", 3: "emojis/happy.PNG",
              4: " emojis/neutral.PNG", 5: "emojis/sad.PNG", 6: "emojis/surpriced.PNG"}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


def bounding_box():
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if (frame is not None):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            show_text[0]=maxindex

        cv2.imshow('Video', cv2.resize(frame, (700, 600), interpolation=cv2.INTER_CUBIC))
        frame2 = cv2.imread(emoji_dist[show_text[0]])
        print('2--vid2')
        if (frame2 is not None):
            pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
        cv2.imshow('Emoji', cv2.resize(pic2, (700, 600), interpolation=cv2.INTER_CUBIC))
        # ret1, jpeg=cv2.imencode('.jpg',frame)
        # ret1, jpeg1=cv2.imencode('.jpg', pic2)
        # return jpeg.tobytes(),jpeg1.tobytes()

        if cv2.waitKey(1) & 0xFF == ord('q'):
             cap.release()
             # cv2.destroyAllWindows()
             break

# def show_vid():
#     print('vid----111')
#     width, height = 800, 600
#     cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#     cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     if not cap1.isOpened():
#         print("cant open the camera1")
#     flag1, frame1 = cap1.read()
#     frame1 = cv2.resize(frame1, (600, 500))
#
#     bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#         prediction = model.predict(cropped_img)
#
#         maxindex = int(np.argmax(prediction))
#         cv2.putText(frame1, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,cv2.LINE_AA)
#         show_text[0] = maxindex
#     if flag1 is None:
#         print("Major error!")
#     elif flag1:
#         global last_frame1
#         last_frame1 = frame1.copy()
#         pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGBA)
#         img = PIL.Image.fromarray(pic)
#         imgtk = ImageTk.PhotoImage(image=img)
#         lmain.imgtk = imgtk
#         lmain.configure(image=imgtk)
#         lmain.after(10, show_vid)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             exit()
bounding_box()