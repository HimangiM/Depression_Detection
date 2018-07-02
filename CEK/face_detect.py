
# coding: utf-8

# In[37]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[38]:


haar_face_cascade = cv2.CascadeClassifier('C:\\Users\\IHI1KOR\\AppData\\Local\\Continuum\\anaconda2\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml')


# In[80]:


def read_face(path):
    test = cv2.imread(path)
    faces = haar_face_cascade.detectMultiScale(test, scaleFactor=1.05, minNeighbors=5)
    print ("Faces", len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(test, (x,y), (x+w, y+h), (0, 255, 0), 1)
        print (x, y, w, h)
    img = test[y:y+h-1, x:x+w-1]
    plt.imshow(img)
    return img


# In[3]:


test1 = cv2.imread("C:/Users/IHI1KOR/Desktop/Work_emotion/CEK+/extended-cohn-kanade-images/cohn-kanade-images/S005/001/S005_001_00000001.png")


# In[4]:


plt.imshow(test1)


# In[29]:


# Begin extracting faces for all files from here


# In[ ]:


directory = "C:/Users/IHI1KOR/Desktop/Work_emotion/CEK+/extended-cohn-kanade-images/cohn-kanade-images"
for file in os.listdir(directory):
    for folder in os.listdir(directory + "/" + file):
        if folder == ".DS_Store":
            continue
        for image in os.listdir(directory + "/" + file + "/" + folder):
            print (image)
            if not os.path.exists(direc + "/" + file + "/" + folder):
                os.makedirs(direc + "/" + file + "/" + folder)
            img = read_face(directory + "/" + file + "/" + folder + "/" + image)
            cv2.imwrite(direc + "/" + file + "/" + folder + "/" + image, img)

