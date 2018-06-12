

# coding: utf-8

# In[1]:


import csv
import os


# In[2]:


for j in os.listdir("./frames_30fps"):
    for csv_file in os.listdir("./frames_30fps/" + str(j)):
        
        _data = csv.reader(open("./frames_30fps/" + str(j) + "/" + csv_file, "r"))
        data = csv.reader(open("./frames_30fps/" + str(j) + "/" + csv_file, "r"))
        data_op = csv.writer(open("./frames_10fps/" + str(j) + "/" + csv_file, "wb"))
        
        print (csv_file)
        _len = len(list(_data)) - 1
        count = (_len / 3)

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

