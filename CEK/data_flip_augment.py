
# coding: utf-8

# In[15]:


import os
import cv2
from PIL import Image


# In[16]:


direc = "/home/intern_eyecare/Emotion data/CEK+/CEK_64_64_augmented_class_duplicate"
directory = "/home/intern_eyecare/Emotion data/CEK+/CEK_flip_augment"
for file in os.listdir(direc):
    if file != "test" and file != "validation":
        for folder in os.listdir(direc + "/" + file):
            print (folder)
            for image in os.listdir(direc + "/" + file + "/" + folder):
                print (image)
                img = cv2.imread(direc + "/" + file + "/" + folder + "/" + image, 0)
                print (img.shape)
                img_ = img.copy()
                img_ = cv2.flip(img, 1)

                cv2.imwrite(directory + "/" + file + "/" + folder + "/" + image, img_)
    #             print (directory + "/" + file + "/" + folder + "/" + image)
    #             cv2.imshow("orig", img)
    #             cv2.imshow("flip", img_)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()


# In[ ]:


direc = "/home/intern_eyecare/Emotion data/CEK+/CEK_64_64_augmented_class_duplicate"

for folder in os.listdir(direc + "/S103"):
    for image in os.listdir(direc + "/S103/" + folder):
        img = cv2.imread(direc + "/S103/" + folder + "/" + image)
        img_ = img.copy()
        img_ = cv2.flip(img, 1)
        cv2.imshow("orig", img)
        cv2.imshow("flip", img_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

