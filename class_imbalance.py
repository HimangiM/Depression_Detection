
# coding: utf-8

# In[ ]:


# {0: 0, 1: 45, 2: 18, 3: 60, 4: 25, 5: 70, 6: 29, 7: 83}

# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

# 1: 2x, 2: 5x, 3: 1/3 add, 4: 3x, 5: none, 6: 3x, 7: None

# In[25]:


import os
import csv
import shutil
from random import shuffle
from shutil import copyfile

# In[26]:


# duplicate name as session_dup


# In[34]:

# duplicate sessions
l_one = []
l_two = []
l_three = []
l_four = []
l_six = []
with open("label_path.csv", "r") as infile:
    infile.readline()
    for line in infile.readlines():
        token = line.strip().split("\t")
        print (token)
        print (token[0][-8:], token[1])
        if token[1] == "1":
            l_one.append(token[0][-8:])
        if token[1] == "2":
            l_two.append(token[0][-8:])
        if token[1] == "3":
            l_three.append(token[0][-8:])
        if token[1] == "4":
            l_four.append(token[0][-8:])
        if token[1] == "6":
            l_six.append(token[0][-8:])


# name = 65
# letter = chr(name)
direc = "/home/himangi/Bosch/CEK/CEK_64_64_augmented_class_duplicate"


# 1: 2x, 2: 5x, 3: 1/3 add, 4: 3x, 5: none, 6: 3x, 7: None

# name = 65
# letter = chr(name)
# print l_one
# for i in l_one:
#     if not os.path.exists(direc + "/" + i + letter):
#         path = direc + "/" + i + letter
#         label = "1"
#         with open("/home/himangi/Bosch/CEK/label_path.csv", 'a') as outfile:
#             print (path + "\t" + label + "\n")
#             row = path + "\t" + label + "\n"
#             outfile.write(row)
#         outfile.close()
#         os.makedirs(direc + "/" + i + letter)
#         for file in os.listdir(direc + "/" + i):
#             copyfile(direc + "/" + i + "/" + file, direc + "/" + i + letter + "/" + file)


# print l_two
# names = [65, 66, 67, 68]
# for n in names:
#     letter = chr(n) 
#     for i in l_two:
#         if not os.path.exists(direc + "/" + i + letter):
#             path = direc + "/" + i + letter
#             label = "2"
#             with open("label_path.csv", 'a') as outfile:
#                 print (path + "\t" + label + "\n")
#                 row = path + "\t" + label + "\n"
#                 outfile.write(row)
#             outfile.close()
#             os.makedirs(direc + "/" + i + letter)
#             for file in os.listdir(direc + "/" + i):
#                 copyfile(direc + "/" + i + "/" + file, direc + "/" + i + letter + "/" + file)
             

# print l_three
# shuffle(l_three)
# sample = l_three[0:20]
# print sample
# sample = ['S116/006', 'S054/004', 'S106/004', 'S046/004', 'S131/010', 'S125/008', 'S128/004', 'S045/004', 'S078/007', 
#         'S071/006', 'S081/008', 'S080/008', 'S088/004', 'S090/006', 'S097/004', 'S032/005', 'S044/006', 'S107/005', 
#         'S061/004', 'S058/006']

# name = 65
# letter = chr(name)
# for i in sample:
#     if not os.path.exists(direc + "/" + i + letter):
#         path = direc + "/" + i + letter
#         label = "3"
#         with open("label_path.csv", 'a') as outfile:
#             print (path + "\t" + label + "\n")
#             row = path + "\t" + label + "\n"
#             outfile.write(row)
#         outfile.close()
#         os.makedirs(direc + "/" + i + letter)
#         for file in os.listdir(direc + "/" + i):
#             copyfile(direc + "/" + i + "/" + file, direc + "/" + i + letter + "/" + file)



# names = [65, 66]
# print l_four
# for n in names:
#     letter = chr(n)
#     for i in l_four:
#         if not os.path.exists(direc + "/" + i + letter):
#             path = direc + "/" + i + letter
#             label = "4"
#             with open("label_path.csv", 'a') as outfile:
#                 print (path + "\t" + label + "\n")
#                 row = path + "\t" + label + "\n"
#                 outfile.write(row)
#             outfile.close()
#             os.makedirs(direc + "/" + i + letter)
#             for file in os.listdir(direc + "/" + i):
                # copyfile(direc + "/" + i + "/" + file, direc + "/" + i + letter + "/" + file)
         
names = [65, 66]
print l_six
for n in names:
    letter = chr(n)
    for i in l_six:
        if not os.path.exists(direc + "/" + i + letter):
            path = direc + "/" + i + letter
            label = "6"
            with open("label_path.csv", 'a') as outfile:
                print (path + "\t" + label + "\n")
                row = path + "\t" + label + "\n"
                outfile.write(row)
            outfile.close()
            os.makedirs(direc + "/" + i + letter)
            for file in os.listdir(direc + "/" + i):
                copyfile(direc + "/" + i + "/" + file, direc + "/" + i + letter + "/" + file)
         


# 1A, 2ABCD, 4AB, 6AB done
# 1A, 2ABCD, 4(AB), 6(AB)
# 3A(1/3)
# # In[36]:


# if not os.path.exists(direc+"_duplicate/S072/005_dup"):
#     os.makedirs(direc + "_duplicate/S072/005_dup")

# shutil.copytree("/home/intern_eyecare/temp", "/home/intern_eyecare/himangi")

