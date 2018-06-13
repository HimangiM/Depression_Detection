
# coding: utf-8

# In[27]:


# 411 samples
# 247 train
# 82 validation
# 82 test

# create separate list of zero and one labeled files, and sample half-half from them
# Don't create half zeroes and half ones, do it random

import os
import random


# In[28]:


test_valid_size = 82 


# In[29]:


file_list = []
zero_names = [302, 303, 304, 305, 307, 310, 312, 313 , 324, 316]
one_names = [335, 325, 330, 346]
rand_test = []
rand_valid = []
for filename in os.listdir("./frames_10fps"):
    if filename != "validation" and filename != "test":
        for file in os.listdir("./frames_10fps/"+filename):
    #         print (file[0:3])
            file_list.append(file)
                

rand_test = random.sample(zero_l, test_valid_size)

files_no_test  = [i for i in file_list if i not in rand_test]

rand_valid = random.sample(files_no_test, test_valid_size)

for i in rand_test:
    os.rename("./frames_10fps/"+str(i[0:3])+"_P/"+str(i), "./frames_10fps/test/"+str(i))    
    
for j in rand_valid:
    os.rename("./frames_10fps/"+str(j[0:3])+"_P/"+str(j), "./frames_10fps/validation/"+str(j))    


# In[25]:


# create test


# In[25]:


# send from test/validation to original directory
for test_file in os.listdir("./frames_10fps/test"):
    print (test_file)
    os.rename("./frames_10fps/test/"+str(test_file), "./frames_10fps/"+str(test_file[0:3])+"_P/"+str(test_file))


# In[26]:


for valid_file in os.listdir("./frames_10fps/validation"):
    print (valid_file)
    os.rename("./frames_10fps/validation/"+str(valid_file), "./frames_10fps/"+str(valid_file[0:3])+"_P/"+str(valid_file))

