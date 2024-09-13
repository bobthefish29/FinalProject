'''Welcome to the project for Iphone VS Price


This project is for price of iphone units from its inception to the present time


price  |--------------------------------
price  |--------------------------------
price  |-----------____0____-----------
price  |---------/-----------\ ----
price  |-------/--------------0----------
price  |-----0--------------------------
Iphone | Iphone 1, iphone 2, iphone 3

.         Future Date = ____


this is what it will look like (lol)

it could be a ---line--- or a ---graph--- 

---than caluate how much it will be in 2,3,5 years---

than have a future date from an input




'''#keep me

#this is where we will be having the imports
import pandas as pd


'''file infomation

1.the 'r' just in case, its so that your computer could read the file if there is an issus
2. that is where we are giving each column a name,


'''

file = pd.read_csv(r'data/SeedUnofficialAppleDataCSV.csv')#1
file.columns = ['model', 'release_os', 'release_date', 'discontinued', 'support_ended','final_os', 'lifespan', 'max_lifespan', 'launch_price']#2


#This function get you a whole list of all the iphones> it returns as "iphone ????"  ???==Number/model
#this could just be used for lables and other stuff.
def getPhoneList():
    #this is droping the null values
    file.dropna(subset='model', inplace=True)
    #this is just lists for random
    phoneList = []
    phoneList2 = []

    #this is removing the "/" in the iphones like iphone 6 / 6s 
    #when this is returning the ./ list it comes ["iphone 6","6s "]
    for i in file['model']:    
        phoneList.append(i.split("/"))

    for i in range(1, len(phoneList)):
        #this is fixing the ["iphone 6","6s "] issue, is checking
        if(len(phoneList[i]) > 1):
            for l in range(0,len(phoneList[i])):
                phoneList2.append(phoneList[i][l])
        else:#this is appending the item after
            phoneList2.append(phoneList[i][0])

    #This is just adding "iphone" to each item
    for i in range(1,len(phoneList2)):
        if(phoneList2[i][0] == " "):
            phoneList2[i] = "iphone" + phoneList2[i]
    return phoneList2


getPhoneList()




#this section is a work in progross, if you want to take ove and try anything please do
priceList = []
priceList2 = []
priceList3 = []


#this is removing the $
for i in file['launch_price']:
    priceList.append(i.replace('$', ''))
    # priceList.append(i.split("/"))
#this is removing the *
for i in priceList:
    priceList2.append(i.replace('*', ''))
#this is removing the / 
for i in priceList2:
    priceList3.append(i.split("/"))


print(priceList3)

# for i in priceList:

    
    
#     #priceList.append(i.replace('$', ''))

#     print(i.replace('$', ' '))
#     # print(i.split("/"))

# for i in priceList:
#     priceList2.append(i.replace('*', ''))
#     # print(i.replace('$', ' '))


# # print(priceList2)




