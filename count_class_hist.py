import os
import csv
import matplotlib.pyplot as plt

l = []

# f_out = open("label_path.csv", "w")
# f_out.write("path\tlabel\n")
# f_out.close()

direc = "/home/himangi/Bosch/CEK/Emotion"
directory = "/home/himangi/Bosch/CEK/CEK_64_64_augmented"

# with open("/home/himangi/Bosch/CEK/label_path.csv", "a") as outfile:
# 	for file in os.listdir(direc):
# 		if file != "test" and file != "validation":
# 			# print (file) 
# 			for folder in os.listdir(direc + "/" + file):
# 				for txtfile in os.listdir(direc + "/" + file + "/" + folder):
# 					print txtfile
# 					f = open(direc + "/" + file + "/" + folder + "/" + txtfile)
# 					label = int(float((f.read()).strip()))
# 					print label
# 					l.append(label)
# 					path = directory + "/" + file + "/" + folder
# 					outfile.write(path + "\t" + str(label) + "\n")
# 					f.close()


# outfile.close()
# plt.hist(l, bins = 13)
# plt.show()


with open("/home/himangi/Bosch/CEK/label_path.csv", "r") as infile:
	infile.readline()
	for line in infile.readlines():
		token = line.strip().split("\t")
		l.append(token[1])

print ("h")
# plt.hist(l, bins = 13)
# plt.show()
print l

d = dict()
d['0'] = d['1'] = d['2'] = d['3'] = d['4'] = d['5'] = d['6'] = d['7'] = 0
for i in l:
	d[i] = d[i]+1

print d