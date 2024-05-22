#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[3]:

import streamlit as st
import cv2
import numpy as np

#set title for the application
st.title("My First Streamlit Application")

st.write("Please Upload Your Image")

#display a file uploader widget
uploaded_image = st.file_uploader("choose an Image..", type = ['jpg', 'jpeg', 'png'])

#if an image is uploaded, display it
if uploaded_image is not None:
    image = np.array(bytearray(uploaded_image.read()), dtype = np.uint8)

    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    #display image
    st.image(uploaded_image, caption = 'BGR Image.', channels = 'BGR')

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #display gray image
    st.image(gray_image, caption = 'Gray Image.', channels = 'GRAY')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (255, 240,0), 2)
    
    st.image(img, channels = 'BGR', caption ='Face Detection')
