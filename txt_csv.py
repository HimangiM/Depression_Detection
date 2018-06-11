
import pandas as pd
import numpy as np
import csv
import pickle as pkl

with open("labels_dict.pkl", "rb") as infile:
    labels_dict = pkl.load(infile)

for i in range(302, 304):
    txt_file1 = "./dataset/USC/"+str(i)+"_P/"+str(i)+"_CLNF_features.txt"
    txt_file2 = "./dataset/USC/"+str(i)+"_P/"+str(i)+"_CLNF_features3D.txt"
    txt_file3 = "./dataset/USC/"+str(i)+"_P/"+str(i)+"_CLNF_gaze.txt"
    txt_file4 = "./dataset/USC/"+str(i)+"_P/"+str(i)+"_CLNF_pose.txt"
    txt_file5 = "./dataset/USC/"+str(i)+"_P/"+str(i)+"_CLNF_AUs.txt"
    csv_file = str(i)+"_P.csv"

    in_txt1 = csv.reader(open(txt_file1, "r"), delimiter = ",")
    in_txt2 = csv.reader(open(txt_file2, "r"), delimiter = ",")
    in_txt3 = csv.reader(open(txt_file3, "r"), delimiter = ",")
    in_txt4 = csv.reader(open(txt_file4, "r"), delimiter = ",")
    in_txt5 = csv.reader(open(txt_file5, "r"), delimiter = ",")

    with open(csv_file, 'wb') as outfile:
        out_csv = csv.writer(outfile)
        l1 = next(in_txt1)
        l2 = next(in_txt2)
        l3 = next(in_txt3)
        l4 = next(in_txt4)
        l5 = next(in_txt5)

        l1.pop(2)
        del l2[0:4]
        del l3[0:4]
        del l4[0:4]
        del l5[0:4]

        combined_row = []
        combined_row = l1 + l2 + l3 + l4 + l5
        combined_row.append("label")
        out_csv.writerow(combined_row)

        for row1, row2, row3, row4, row5 in zip(in_txt1, in_txt2, in_txt3, in_txt4, in_txt5):
            combined_row = []
            row1.pop(2)
            del row2[0:4]
            del row3[0:4]         
            del row4[0:4]         
            del row5[0:4]         
            combined_row = row1 + row2 + row3 + row4 + row5
            combined_row.append(labels_dict[i])
            print combined_row  
            out_csv.writerow(combined_row)




    

