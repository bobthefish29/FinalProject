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
import matplotlib.pyplot as plt
import numpy as np


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
    tfile = file.dropna(subset='model', inplace=False, axis='index')
    #this is just lists for random
    phoneList = []
    phoneList2 = []

    #print(tfile)

    #this is removing the "/" in the iphones like iphone 6 / 6s 
    #when this is returning the ./ list it comes ["iphone 6","6s "]
    for i in tfile['model']:    
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


# print('There are',len(getPhoneList()),'Iphones\n\n')


#this section is a work in progross, if you want to take ove and try anything please do

#this is taking in the infomration from the file, than it will return a whole list of the phone and the prices for that list
#Phone          FirstPrice   Carrier price
#Iphone      [325,329]          894
# iphone 3  [235,232]           null
#
#
def getPriceList():
    #['launch_price'].dropna(inplace=False, axis='index')

    #print('\n\t--In the price function---\n')
    tempFile = file
    result_list = []
    realFinal = []

    #this is adding all the values and stuff together
    #what its doing it looking at the model row, if the value is null than it would add the lanch price to a list of the item before
    #if its not null than it would add the value to the list
    for i, r in tempFile.iterrows():
        if pd.isnull(r['model']):
            # Add the launch_price to the last non-null model's list
            result_list[-1][1].append(r['launch_price'])
        else:
            # Create a new entry for the non-null model
            result_list.append([r['model'], [r['launch_price']]])





    #this is for a lot ngl, its first looping through each row
    for i in range(1,len(result_list)):
        #this is for the first row, if there is a "/" it will spice it, if not it will just add it to its own list
        #if it does find a "/" it replace it with a space
        result_list[i][0] = result_list[i][0].split("/")

        #this is than looking if the item in the firs list has a " " than it puts iphone.
        for l in range(0, len(result_list[i][0])):
            if result_list[i][0][l].startswith(" "):
                #print(result_list[i][0][l])
                result_list[i][0][l] = "iphone" + result_list[i][0][l]
                # print(result_list[i][0][l])
        #this is looking at the price list, its each item in the price list
        #if it has a "*" than it removes it, if it has a "$" it removes it
        for l in range(0,len(result_list[i][1])):
            # print(result_list[i][1][l].replace('$', ''))
            result_list[i][1][l] = result_list[i][1][l].replace('*', '')
            result_list[i][1][l] = result_list[i][1][l].replace('$', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Plus:', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Max:', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Mini:', '')

    for i in range(1,len(result_list)):
        if len(result_list[i][0]) == 1:#if the first list only has one item than it will do this
            #print(f"---only one[1][]: {result_list[i]}--")
            if len(result_list[i][1]) > 1:
                #print(f"more than 1[1][2]: {result_list[i][1]}")
                #realFinal.append([result_list[i][0][0],[result_list[i][1][0], result_list[i][1][1]]])
                realFinal.append([result_list[i][0][0],result_list[i][1][0], result_list[i][1][1]])
                #print([result_list[i][0][0],[result_list[i][1][0], result_list[i][1][1]]])
            else:
                #print(f"less than 1[1][1]: {result_list[i][1]}")
                realFinal.append([result_list[i][0][0],result_list[i][1][0]])
                #print([result_list[i][0][0],[result_list[i][1][0]]])
            #print('\n')
        else:
            #print(f"---more than one[2][]: {result_list[i]}")
            if len(result_list[i][1]) > 2:
                #print(f"more than two[2][4]: {result_list[i][l]}")
                for l in range(0, len(result_list[i][0])):
                    #print(f"loop {l}")
                    if l == 0:
                        realFinal.append([result_list[i][0][l],result_list[i][1][0],result_list[i][1][1]])
                        #print([result_list[i][0][l],[result_list[i][1][0],result_list[i][1][1]]])
                    else:
                        realFinal.append([result_list[i][0][l],result_list[i][1][2],result_list[i][1][3]])
                        #print([result_list[i][0][l],[result_list[i][1][2],result_list[i][1][3]]])
                #print('\n')
            else:
                for l in range(0, len(result_list[i][0])):
                    #print(f"loop {l}")
                    if l == 0:
                        realFinal.append([result_list[i][0][l],result_list[i][1][0]])
                        #print([result_list[i][0][l],[result_list[i][1][0]]])
                    else:
                        realFinal.append([result_list[i][0][l],result_list[i][1][1]])
                        #print([result_list[i][0][l],[result_list[i][1][1]]])

    #this is converting the whole thing i just made into a data frame
    df = pd.DataFrame(realFinal, columns=['iphone', 'price', 'plane'])

    df['price'] = df['price'].str.split("/")
    df['plane'] = df['plane'].str.split("/")

    return df
        

priceDf = getPriceList()

# print(priceDf)


'''
#this is the first graph i was working on, its super basic and i was just having fun
y = [int(x[0]) for x in priceDf['price']] #this is the first price list SO:[122,233]  ITs the 122
y1 = [int(x[1]) for x in priceDf['price']] #this is the next price list SO:[122,233]  ITs the 233
x = [x for x in priceDf['iphone']] # this is the iphones, thats why its the lables

# # Create a figure and axes
fig, ax = plt.subplots()

# # Plot the data
ax.plot(x, y)#this is doing iphone on x an dthe valeus on the y
ax.plot(x, y1)#this is doing the iphone on x an the valeus on y

# # Add a title and labels
ax.set_title("Sine Wave")
ax.set_xlabel("x")
plt.xticks(rotation=90)
ax.set_ylabel("y")
plt.show()
'''

##this is how we can make a bestFitLine
#y = (m*x) + b
#m = slopt
#x = its the line
#b = its the line itself


'''
from statistics import mean
import numpy as np

# int(x[1]) for x in priceDf['price']

y = [int(x[0]) for x in priceDf['price']] #this is the first price list SO:[122,233]  ITs the 122
x = [int(x[0]) for x in priceDf['price']] #this is the next price list SO:[122,233]  ITs the 233
iphone = [x for x in priceDf['iphone']] # this is the iphones, thats why its the lables



# xs = np.array(iphone, dtype=np.strings)
#xs = np.array(int(x[0]) for x in priceDf['price'])
# ys = np.array(int(x[1]) for x in priceDf['price'])
xs = np.array(x, dtype=np.float64)
ys = np.array(y, dtype=np.float64)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)
# print(m,b)
regression_line = [(m*x)+b for x in xs]
from matplotlib import style
style.use('ggplot')
plt.scatter(iphone,ys,color='#003F72',label='price')
plt.plot(iphone,regression_line,label='regression line')
# plt.plot(iphone,ys,label='regression line')
# plt.plot(xs,ys,label='regression line')
# plt.tick_params(tick1On=xs.all(),tick2On=ys.all())
plt.xticks(iphone,rotation=90)
plt.legend(loc=4)
plt.show()
predict_x = 7
predict_y = (m*predict_x)+b
'''

# x_values = np.array([0, 1, 2, 3, 4])
# y_values = np.array([2, 3, 5, 7, 11])

# # Fit a linear polynomial (degree 1)
# coefficients = np.polyfit(x_values, y_values, 1)
# slope, intercept = coefficients

# print(f"Slope: {slope}")
# print(f"Intercept: {intercept}")




from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style



# Extracting data
# y = [int(x[0]) for x in priceDf['price']]  # First price list
# x = [int(x[0]) for x in priceDf['price']]  # first price list

def makeBasicTable(xlist,ylist):
    #just a basic making of a iphone list
    iphone = [x for x in priceDf['iphone']]# iPhone labels

    #this is taking the values that were being inputed and turing them into lists
    xs = np.array(xlist, dtype=np.float64)
    ys = np.array(ylist, dtype=np.float64)

    # Function to calculate slope and intercept
    def best_fit_slope_and_intercept(xs, ys):
        print(((mean(xs) * mean(ys)) - mean(xs * ys)))
        print(((mean(xs) * mean(xs)) - mean(xs * xs)))
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
        #this is the y int value
        b = mean(ys) - m * mean(xs)
        return m, b

    # # Calculate slope and intercept
    m, b = best_fit_slope_and_intercept(xs, ys)


    # Plotting
    style.use('ggplot')

    #so this is the x values, its the lengh of the ipones[0,1,2,3,4,5,6,7,8,9]
    x_values = np.arange(len(iphone))

    # Calculate the line of best fit, so the way polyfit is working is that it take in two list numbers
    #it than does some math on the numbers inputed and get out just 2 values, the first value is the slope
    #the secoend vlaue is the intersept
    slope, intercept = np.polyfit(x_values, ys, 1)

    #this is a list of all the values, its basicly doing the slope value * each item in the lsit + the intersept value
    best_fit_line = slope * x_values + intercept

    # print(best_fit_line)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(iphone, xs, marker='o', label='iPhone Prices')
    plt.plot(iphone, best_fit_line, color='green', linestyle='--', label='Best Fit Line')

    # Add labels and title
    plt.xlabel('iPhone Models')
    plt.ylabel('Price (USD)')
    plt.title('iPhone Prices by Model')
    plt.xticks(rotation=45)#Rotate x-axis labels for better readability
    plt.legend()

    #this is adding the numbers to each point
    for i, price in enumerate(xs):
        plt.annotate(f'${price}', (iphone[i], xs[i]), textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()
    #Ending of the function

def makeTableWithYear(xlist,ylist, year):
    #just a basic making of a iphone list
    iphone = [x for x in priceDf['iphone']]# iPhone labels

    #this is taking the values that were being inputed and turing them into lists
    xs = np.array(xlist, dtype=np.float64)
    ys = np.array(ylist, dtype=np.float64)

    i = 0
    #FV=PV(1+i)n
    xs = pd.DataFrame(xs)
    ys = pd.DataFrame(ys)
    next_value = (xs.iloc[-1] + ys.iloc[-1])/2
    print(xs)
    print(next_value)
    # while i < year:
    #     print(i)
    #     i = i + 1


    # Function to calculate slope and intercept
    # def best_fit_slope_and_intercept(xs, ys):
    #     print(((mean(xs) * mean(ys)) - mean(xs * ys)))
    #     print(((mean(xs) * mean(xs)) - mean(xs * xs)))
    #     m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    #     #this is the y int value
    #     b = mean(ys) - m * mean(xs)
    #     return m, b

    # # # Calculate slope and intercept
    # m, b = best_fit_slope_and_intercept(xs, ys)


    # Plotting
    style.use('ggplot')

    #so this is the x values, its the lengh of the ipones[0,1,2,3,4,5,6,7,8,9]
    x_values = np.arange(len(iphone) + year)

    print(x_values)
    # Calculate the line of best fit, so the way polyfit is working is that it take in two list numbers
    #it than does some math on the numbers inputed and get out just 2 values, the first value is the slope
    #the secoend vlaue is the intersept
    slope, intercept = np.polyfit(x_values, ys, 1)

    #this is a list of all the values, its basicly doing the slope value * each item in the lsit + the intersept value


    best_fit_line = slope * x_values + intercept

    # print(best_fit_line)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(iphone, xs, marker='o', label='iPhone Prices')
    plt.plot(iphone, best_fit_line, color='green', linestyle='--', label='Best Fit Line')

    # Add labels and title
    plt.xlabel('iPhone Models')
    plt.ylabel('Price (USD)')
    plt.title('iPhone Prices by Model')
    plt.xticks(rotation=45)#Rotate x-axis labels for better readability
    plt.legend()

    #this is adding the numbers to each point
    for i, price in enumerate(xs):
        plt.annotate(f'${price}', (iphone[i], xs[i]), textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()
    #Ending of the function





# y = [int(x[0]) for x in priceDf['price']]  # First price list
# x = [int(x[0]) for x in priceDf['price']]  # first price list
# makeBasicTable(x,y)


y = [int(x[0]) for x in priceDf['price']]  # First price list
x = [int(x[0]) for x in priceDf['price']]  # first price list
makeTableWithYear(x,y,9)



# y = [int(x[1]) for x in priceDf['price']]  # First price list
# x = [int(x[1]) for x in priceDf['price']]  # first price list
# makeTable(x,y)





# predict_x = 7
# predict_y = (m*predict_x)+b

# plt.scatter(xs,ys,color='#003F72',label='data')
# plt.plot(xs, regression_line, label='regression line')
# plt.legend(loc=4)
# plt.show()






# priceDf = getPriceList()

# print(len(priceDf))


#this is removing the $
# for i in file['launch_price']:
#     print(i)






#     # priceList.append(i.split("/"))
# #this is removing the *
# for i in lstpriceList:
#     priceList2.append(i.replace('*', ''))
# #this is removing the / 
# for i in priceList2:
#     priceList3.append(i.split("/"))

# lstpriceList = priceList3
# lstpriceList.pop(0)
# print(lstpriceList)







# for i in priceList:

    
    
#     #priceList.append(i.replace('$', ''))

#     print(i.replace('$', ' '))
#     # print(i.split("/"))

# for i in priceList:
#     priceList2.append(i.replace('*', ''))
#     # print(i.replace('$', ' '))


# # print(priceList2)




