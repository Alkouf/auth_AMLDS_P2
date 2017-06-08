import numpy as np
with open("extras/silhouette_analysis_results.txt","rb") as f:
    data =f.readlines()
    f.close()

print(data)

coefficient=[]

for row in data:
    print(row)
    print("____________________")
    coefficient.append(float(row.split(":")[1].replace("\n","")))


coef =np.array(coefficient)

print(coef.argsort())

print(coef[coef.argsort()])