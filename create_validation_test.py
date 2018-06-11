
# coding: utf-8

# In[33]:


# 411 samples
# 247 train
# 82 validation
# 82 test

# create separate list of zero and one labeled files, and sample half-half from them

import os
import random


# In[38]:


zero_l = []
one_l = []
zero_names = [302, 303, 304, 305, 307, 310, 312, 313 , 324]
one_names = [335, 325, 330, 346]
rand_zeroes = []
rand_ones = []
for filename in os.listdir("./frames_30fps"):
    for file in os.listdir("./frames_30fps/"+filename):
#         print (file[0:3])
        if int(file[0:3]) in zero_names:
            zero_l.append(file)
        else:
            one_l.append(file)

print (len(zero_l), len(one_l))
rand_zeroes = random.sample(zero_l, 42)
rand_ones = (random.sample(one_l, 42))
for i in rand_zeroes:
    os.rename("./frames_30fps/"+str(i[0:3])+"_P/"+str(i), "./frames_30fps/test/"+str(i))    
for j in rand_ones:
    os.rename("./frames_30fps/"+str(j[0:3])+"_P/"+str(j), "./frames_30fps/test/"+str(j))    


# In[25]:


# create test


# In[37]:


# send from test/validation to original directory
for test_file in os.listdir("./frames_30fps/test"):
    print (test_file)
    os.rename("./frames_30fps/test/"+str(test_file), "./frames_30fps/"+str(test_file[0:3])+"_P/"+str(test_file))

