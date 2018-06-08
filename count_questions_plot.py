
# coding: utf-8

# In[15]:


import csv
import os
import matplotlib.pyplot as plt


# In[23]:


count = 0
_max = 0
x = []
y = []
_sum= 0
for filename in os.listdir("./frames"):
    for csv_file in os.listdir("./frames/" + filename):
        count = count + 1
        x.append(count)
        data = csv.reader(open("./frames/" + filename + "/" + csv_file, "r"))
        _data = csv.reader(open("./frames/" + filename + "/" + csv_file, "r"))
        _data_ = csv.reader(open("./frames/" + filename + "/" + csv_file, "r"))

#         print (filename+"/"+ csv_file, len(list(_data)))
#         _max = max(_max, int(len(list(data))))
        if len(list(data)) <= 1000:
            _sum = _sum + len(list(_data))
        y.append(len(list(_data_)))

print (_sum/count)
plt.scatter(x, y)
plt.show()

