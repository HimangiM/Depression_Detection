import csv
import pandas as pd

l = [418, 440, 451, 458, 472, 483]

for k in l:
    data_trans = csv.reader(open("./dataset/USC/"+str(k)+"_P/"+str(k)+"_TRANSCRIPT.csv", "r"))
    ques_list = ["why", "how", "what", "when", "which", "where", "who"]

    header2 = next(data_trans)

    total_time = 0
    num_frames = 0
    count = 0
    first_ques = 0
    ques_start_time = 0
    ques_end_time = 0
    for row in data_trans:
        if (row != []):
            token = row[0].strip().split("\t")
            if len(token) == 4:
                token_value = token[3].strip().split(" ")
                token_ques = token_value[0].strip().split("'")
            # print token_ques
            # # print token_ques[0]
            if token_ques[0] in ques_list and token[2] == "Ellie":
                if first_ques == 0:
                    first_ques = 1
                    ques_start_time = token[0]
                    continue
                else:
                    ques_old_start_time = ques_start_time
                    ques_old_end_time = ques_end_time
                    ques_start_time = token[0]

                    start_frame = round(float(ques_old_start_time), 1)
                    if float(start_frame) > float(ques_old_start_time):
                        start_frame = float(start_frame) - 0.1
                    end_frame = round(float(ques_old_end_time), 1)
                    if float(end_frame) < float(ques_old_end_time):
                        end_frame = float(end_frame) + 0.1
                    print (ques_old_start_time, start_frame, ques_old_end_time, end_frame)
                    total_time = total_time + (end_frame-start_frame)


                data_features = csv.reader(open("./dataset/USC/session_feature/"+str(k)+"_P.csv", "r"))
                count = count + 1
                data_frame = csv.writer(open("./frames_30fps/"+ str(k) + "_P/" + str(k) + "_P"+str(count)+".csv", "w", newline=''))
    #             next(data_features)
                data_frame.writerow(next(data_features))
                for row in data_features:
                    if float(row[1]) >= float(start_frame) and float(row[1]) <= float(end_frame):
                        num_frames = num_frames + 1
                        data_frame.writerow(row)

            ques_end_time = token[1]

        # For last question
        start_frame = round(float(ques_start_time), 1)
        if float(start_frame) > float(ques_start_time):
            start_frame = float(start_frame) - 0.1

        ques_end_time = 0
        _data_trans = csv.reader(open("./dataset/USC/"+str(k)+"_P/"+str(k)+"_TRANSCRIPT.csv", "r"))

        for _row in reversed(list(_data_trans)):
            _token = _row[0].strip().split("\t")

            if _token[2] == "Participant" and ques_end_time == 0:
                ques_end_time = _token[1]
            elif _token[2] == "Ellie" and _token[0] != ques_start_time:
                ques_end_time = 0

            if _token[0] == ques_start_time:
                break

        end_frame = round(float(ques_end_time), 1)
        if float(end_frame) < float(ques_end_time):
            end_frame = float(end_frame) + 0.1
        print (ques_start_time, start_frame, ques_end_time, end_frame)
        total_time = total_time + (end_frame-start_frame)


        count = count + 1
        data_features = csv.reader(open("./dataset/USC/session_feature/"+str(k)+"_P.csv", "r"))
        data_frame = csv.writer(open("./frames_30fps/"+ str(k) + "_P/" + str(k) + "_P"+str(count)+".csv", "w", newline=''))
    #     next(data_features)
        data_frame.writerow(next(data_features))
        for row in data_features:
            if float(row[1]) >= float(start_frame) and float(row[1]) <= float(end_frame):
                num_frames = num_frames + 1
                data_frame.writerow(row)

    #     print (total_time)
        print (k, num_frames)
