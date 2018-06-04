import csv
import pandas as pd

data_trans = csv.reader(open("./dataset/USC/303_P/303_TRANSCRIPT.csv", "r"))
ques_list = ["why", "how", "what", "when", "where","who"]

header2 = next(data_trans)

count = 0
for row in data_trans:
	token = row[0].strip().split("\t")
	token_value = token[3].strip().split(" ")
	token_ques = token_value[0].strip().split("'")
	# print token_ques
	# # print token_ques[0]
	if token_ques[0] in ques_list:
		ques_start_time = token[0]
		ques_end_time = token[1]
		
		start_frame = round(float(ques_start_time), 1)
		if float(start_frame) > float(ques_start_time):
			start_frame = float(start_frame) - 0.1
		end_frame = round(float(ques_end_time), 1)
		if float(end_frame) < float(ques_end_time):
			end_frame = float(end_frame) + 0.1

		print token[0], start_frame, token[1], end_frame	

		data_features = csv.reader(open("./dataset/USC/303_P.csv", "r"))
		count = count + 1
		data_frame = csv.writer(open("./frames/303_P"+str(count)+".csv", "wb"))
		data_frame.writerow(next(data_features))
		for row in data_features:
			if float(row[1]) >= float(start_frame) and float(row[1]) <= float(end_frame):
				data_frame.writerow(row)


