import csv
for i in range(1, 2):
    txt_file = csv.reader(open("./frames/303_P/303_P"+str(i)+".csv", "r"))
    out_file = csv.writer(open("30_P"+str(i)+".csv", "w", newline=''))
    l = next(txt_file)
    l.append("label")
#   print (l)
    out_file.writerow(l)
    for row in txt_file:
        combined_row = []
        combined_row = row
        combined_row.append("1")
        out_file.writerow(combined_row)
        
