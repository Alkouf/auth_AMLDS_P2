import csv
with open("data/frames.csv", "rb") as f:
    csvfile = csv.reader(f, delimiter=",",quotechar="|")



print(type(csvfile))

result=[]

for row in csvfile:
    result.append(row)


print(result[0])