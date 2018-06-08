
# coding: utf-8

# In[2]:


import csv


# In[41]:


_data = csv.reader(open("./frames_30fps/300_P/300_P1.csv", "r"))
data = csv.reader(open("./frames_30fps/300_P/300_P1.csv", "r"))
data_op = csv.writer(open("./frames_10fps/300_P/300_P1.csv", "w", newline = ''))

count = int(len(list(_data)) / 3)

data_op.writerow(next(data))
# next(data)
for row in data:
    print ("here")
#     print (row[i])
    if count > 0:
        row2 = next(data)
        row3 = next(data)
        
        l = []
        l.append(row[0])
        l.append(row[1])
        l.append(row[2])
        
        for i in range(3, len(row)):
                l.append(round(((float(row[i]) + float(row2[i]) + float(row3[i]))/3),3))  

        data_op.writerow(l)
        print (l, len(l), count)
        count = count - 1
    else:
        data_op.writerow(row)
        
        

