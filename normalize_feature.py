
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import csv
from scipy import stats
import os
import math


# In[67]:


# print (data)
# for filename in os.listdir("./frames_10fps/original"):
l = ["307_P", "310_P", "312_P", "313_P", "316_P", "324_P", "325_P", "330_P", "335_P", "346_P"]
for filename in l:
    for csvfiles in os.listdir("./frames_10fps/original/"+str(filename)):
    #     print (csvfiles)
        f_in = open("./frames_10fps/original/" + str(filename) + "/" + str(csvfiles), "r")
        f_out = open("./frames_10fps/normalized/" + str(filename) + "/" + str(csvfiles), "w", newline = '')
        data = pd.read_csv("./frames_10fps/original/" + str(filename) + "/" + str(csvfiles), sep = ",")

    #     print (csvfiles, data.iloc[1, idx+3])
        txt_file = csv.reader(f_in)
        outfile = csv.writer(f_out)

        outfile.writerow(next(txt_file))
        print (csvfiles)
        # print (data.iloc[0:5,3:])

        new_ = data.iloc[0:,3:-1]
        label = data.iloc[1, -1]
        print ("label")
        print (label)
        result = (stats.zscore(new_, axis = 0, ddof = 0))

    #     print (len(new_), len(result))
        print (new_.shape, result.shape)
        next(txt_file)
        for row, i in zip(txt_file, result):
    #         print (row, i)
            combined_row = []
            combined_row = row[0:3]
            combined_row = combined_row + list(i)
            combined_row.append(label)
    #         print ("Here")
    #         print (combined_row)
            for k in range(0, len(combined_row)):
                if k!=0 and k!=1 and k!=2:
                    if math.isnan(combined_row[k]):
                        print ("nan found")
    #                     print (combined_row.index(val))
    #                     idx = combined_row.index(val)
                        idx = k
    #                     print (idx+3, k)
                        combined_row[k] = int(data.iloc[1, k])
    #         print (combined_row)

            outfile.writerow(combined_row)

        f_in.close()
        f_out.close()

