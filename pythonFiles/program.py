import csv
import pandas as pd



file = pd.read_csv("data/SeedUnofficialAppleDataCSV.csv")

#when you want to view different data you would have to do the 'unnamed: #' because the data is poop. 
# It is Operating system support(model),relesed OS,Data, discontinued,support end,Final os,lifespan,min,price 
# 
file.dropna(subset='Operating system support', inplace=True)

# print(file['Operating system support'])

phoneList = []
phoneList2 = []

for i in file['Operating system support']:
    
    phoneList.append(i)


for i in range(0,len(phoneList)-1 ):

    print(phoneList[i+1].split("/"))
    phoneList2.append(phoneList[i+1].split("/"))

print(phoneList2[8][1])
# print(file['Unnamed: 2'])


# newFile = pd.DataFrame(file)

# newFile.dropna()
# print(newFile)

# print(file['Unnamed: 8'])

#print(file['Operating system support'])



#this is one way of doint it, it works in a way but not the way i want
# with open('data/SeedUnofficialAppleDataCSV.csv') as file:
#     reader = csv.reader(file)
#         # print(file)
#         # reader = csv.DictReader(file, dialect='excel-tab')
#     reader.dropna()

#     for r in reader:

#         print(r)
        
        