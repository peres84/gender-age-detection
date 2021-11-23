from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import tensorflow as tf
import cv2

# load model
age_model = tf.keras.models.load_model('../data/age_model_16_11_2021.h5')
gender_model = tf.keras.models.load_model('../data/gender_model_16_2021.h5')

#decoding 
classes = ['F','M']
decoding = {0:'0-2', 1:'4-6', 2:'8-13',3:'15-20',4:'25-32',5:'38-43',6:'48-53',7:'60+'}



def frame_predict(array):

    #processing images size 
    face_crop = cv2.resize(array, (128,128))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

     # apply gender detection on face
    age_prediction = age_model.predict(face_crop) 
    gender_prediction = gender_model.predict(face_crop)[0]
    
    #decoding 
    index = np.argmax(age_prediction)
    decoding = {0:'0-2', 1:'4-6', 2:'8-13',3:'15-20',4:'25-32',5:'38-43',6:'48-53',7:'60+'}
    
    gen = "M" if gender_prediction[0] > 0.5 else "F"

    return str(decoding[index])+" "+str(gen)




