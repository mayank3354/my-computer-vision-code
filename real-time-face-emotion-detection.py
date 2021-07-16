# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 03:38:02 2021

@author: dell
"""

import numpy as np
import cv2
import face_recognition
from keras.preprocessing import image
from keras.models import model_from_json



webcam_video_stream = cv2.VideoCapture(0)

face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')

emotions_label = ('angry','disgust' ,'fear','happy', 'sad' , 'surprise', 'neutral' )




all_face_locations = []
while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    for index , current_face_location in enumerate(all_face_locations):
         top_pos , right_pos, bottom_pos, left_pos = current_face_location
         top_pos = top_pos * 4
         right_pos = right_pos * 4
         bottom_pos = bottom_pos * 4
         left_pos = left_pos * 4
         
         print('Found face {} at top:{}, right:{}, bottom:{}, left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
         current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        # current_face_image = cv2.GaussianBlur(current_face_image, (99,99), 30)
        # current_frame[top_pos:bottom_pos,left_pos:right_pos] = current_face_image
         cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255), 2)
         #Preprocess input , convert it to a image asthe data in dateset
         #convert to grayscale
         current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
         # Resize to 48*48
         currrent_face_image = cv2.resize(current_face_image, (48, 48))
         #Convert the PIL into 3d numpy array
         img_pixels = image.img_to_array(current_face_image)
         #Expand the shape of an array into single row multiple columns
         img_pixels  = np.expand_dims(img_pixels, axis = 0)
         #Pixel are in range of [0,255], normalize all pixel into scale of [0, 1]
         img_pixels /= 255
         
         #do prediction using model, get the prediction values for alll 7 expressions
         exp_predictions = face_exp_model.predict(img_pixels)
         #find max indexed prediction value (0 to 7)
         max_index = np.argmax(exp_predictions[0])
         #get corresponding emotion from emotion label
         emotion_label = emotions_label[max_index]
         
         #display the name as text in image
         font = cv2.FONT_HERSHEY_DUPLEX
         cv2.putText(current_frame, emotion_label, (left_pos, bottom_pos), font, 0.5, (255,255,255), 1)
         
         
         
         
    cv2.imshow("Webcam Video", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
             break
         
            
webcam_video_stream.release()
cv2.destroyAllWindows()