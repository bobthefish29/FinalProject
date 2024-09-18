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

file = pd.read_csv(r'data/SeedUnofficialAppleDataCSV.csv', encoding="UTF-8")#1
file.columns = ['model', 'release_os', 'release_date', 'discontinued', 'support_ended','final_os', 'lifespan', 'max_lifespan', 'launch_price']#2




#so the next thing i need to look into is getting the dates


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

#this is where i am playing around with dates
def getRDateList():
    tfile = file.dropna(subset='release_date', inplace=False, axis='index')

    date_object = []
    for i in tfile['release_date']:
        date_object.append(i.replace('�', ' '))


    for i in range(1,len(date_object)):
        # i.split("/")
        print('before',date_object[i])
        # date_object[i] = date_object[i].split("/")
        print(date_object[i])


        # if(len(date_object[i]) > 1):
        #     for l in range(0,len(date_object[i])):
        #         date_object[i].append(date_object[i][l])
        # print('\nafter',date_object[i],'\n')


    print(len(date_object))

    

    # print(tfile['release_date'].replace('June', ' hi'))
    #�
    date_object2 = pd.to_datetime(date_object, format="%B %d, %Y", errors='coerce')

    print(date_object2)

    # print(tfile['release_date'])

# getRDateList()

# print('There are',len(getPhoneList()),'Iphones\n\n')


#this section is a work in progross, if you want to take ove and try anything please do

#this is taking in the infomration from the file, than it will return a whole list of the phone and the prices for that list
#Phone          FirstPrice   Carrier price
#Iphone      [325,329]          894
# iphone 3  [235,232]           null
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



#this is the one that will include the price
def getPriceListWithYear():
    #['launch_price'].dropna(inplace=False, axis='index')

    #print('\n\t--In the price function---\n')
    tempFile = file
    result_list = []
    realFinal = []
    date_object = []


    #this is adding all the values and stuff together
    #what its doing it looking at the model row, if the value is null than it would add the lanch price to a list of the item before
    #if its not null than it would add the value to the list
    for i, r in tempFile.iterrows():
        if pd.isnull(r['model']):
            # Add the launch_price to the last non-null model's list
            result_list[-1][1].append(r['launch_price'])
        else:
            # Create a new entry for the non-null model
            result_list.append([r['model'], [r['launch_price']],r["release_date"]])
            # date_object.append(r["release_date"])
        
    # dateTemp = file.dropna(subset='release_date', inplace=False, axis='index')
    # for i in range(1,len(dateTemp)):
    #     date_object.append(dateTemp)
    #     #date = pd.to_datetime(r["release_date"], format="%B %d, %Y", errors='coerce')
    #     # date_object.append(pd.to_datetime(i, format="%B %d, %Y", errors='coerce'))

    # print(len(date_object))
    # print(len(result_list))




    #this is for a lot ngl, its first looping through each row
    for i in range(1,len(result_list)):
        #this is for the first row, if there is a "/" it will spice it, if not it will just add it to its own list
        #if it does find a "/" it replace it with a space
        result_list[i][0] = result_list[i][0].split("/")


        result_list[i][2] = result_list[i][2].split("/")
        #this is than looking if the item in the firs list has a " " than it puts iphone.
        for l in range(0, len(result_list[i][0])):
            if result_list[i][0][l].startswith(" "):
                #print(result_list[i][0][l])
                result_list[i][0][l] = "iphone" + result_list[i][0][l]
            #this is removing the thing
            result_list[i][2][0] = result_list[i][2][0].replace('�', ' ')
        #this is looking at the price list, its each item in the price list
        #if it has a "*" than it removes it, if it has a "$" it removes it
        for l in range(0,len(result_list[i][1])):
            # print(result_list[i][1][l].replace('$', ''))
            result_list[i][1][l] = result_list[i][1][l].replace('*', '')
            result_list[i][1][l] = result_list[i][1][l].replace('$', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Plus:', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Max:', '')
            result_list[i][1][l] = result_list[i][1][l].replace('Mini:', '')

        if len(result_list[i][2]) == 2:
            result_list[i][2][1] = result_list[i][2][1].replace(' (12 Mini)', '')
            result_list[i][2][1] = result_list[i][2][1].replace(' (12 Pro Max)', '')
            result_list[i][2][1] = result_list[i][2][1].replace(' November', 'November')


        result_list[i][2][0] = result_list[i][2][0].replace(' (12 Pro)', '')
        result_list[i][2][0] = result_list[i][2][0].replace(' (12)', '')

        result_list[i][2][0] = result_list[i][2][0].replace('2020 ', '2020')





        


    # print('for\n\n\n\n\n')
    # for i in range(0, len(result_list)):
    #     print(result_list[i])
    # print('\n\n\n\n\n')


    for i in range(1,len(result_list)):
        if len(result_list[i][0]) == 1:#if the first list only has one item than it will do this
            #print(f"---only one[1][]: {result_list[i]}--")
            if len(result_list[i][1]) > 1:
                #print(f"more than 1[1][2]: {result_list[i][1]}")
                #realFinal.append([result_list[i][0][0],[result_list[i][1][0], result_list[i][1][1]]])
                realFinal.append([result_list[i][0][0],result_list[i][1][0],result_list[i][1][1],result_list[i][2][0]])
                #print([result_list[i][0][0],result_list[i][1][0],result_list[i][1][1],result_list[i][2][0]])
            else:
                #print(f"less than 1[1][1]: {result_list[i][1]}")
                realFinal.append([result_list[i][0][0],result_list[i][1][0],'nun',result_list[i][2][0]])
                #print([result_list[i][0][0],result_list[i][1][0],'null',result_list[i][2]])
            #print('\n')
        else:
            #print(f"---more than one[2][]: {result_list[i]}")
            if len(result_list[i][1]) > 2:
                #print(f"more than two[2][4]: {result_list[i][l]}")
                for l in range(0, len(result_list[i][0])):
                    #print(f"loop {l}")
                    if l == 0:
                        realFinal.append([result_list[i][0][l],result_list[i][1][0],result_list[i][1][1],result_list[i][2][0]])
                        #print([result_list[i][0][l],[result_list[i][1][0],result_list[i][1][1]],result_list[i][2]])
                    else:
                        realFinal.append([result_list[i][0][l],result_list[i][1][2],result_list[i][1][3],result_list[i][2][0]])
                        #print([result_list[i][0][l],[result_list[i][1][2],result_list[i][1][3]],result_list[i][2][0]])
                #print('\n')
            else:
                for l in range(0, len(result_list[i][0])):
                    #print(f"loop {l}")
                    if l == 0:
                        realFinal.append([result_list[i][0][l],result_list[i][1][0],'nun',result_list[i][2][0]])
                        #print([result_list[i][0][l],result_list[i][1][0],'nun',result_list[i][2][0]])
                    else:
                        if(result_list[i][0][l] == 'iphone 12 Mini' or result_list[i][0][l] == 'iphone 12 Pro Max'):
                            #print([result_list[i][0][l],[result_list[i][1][1]],'nun',result_list[i][2][1]])
                            realFinal.append([result_list[i][0][l],result_list[i][1][1],'nun',result_list[i][2][1]])
                        else:
                            #print([result_list[i][0][l],[result_list[i][1][1]],'nun',result_list[i][2][0]])
                            realFinal.append([result_list[i][0][l],result_list[i][1][1],'nun',result_list[i][2][0]])

    #this is converting the whole thing i just made into a data frame

    
    df = pd.DataFrame(realFinal, columns=['iphone', 'price', 'plane','year'])

    df['year'] = df['year'].replace('', '')
    df['year'] = pd.to_datetime(df['year'], format="%B %d, %Y", errors='coerce')


    df['price'] = df['price'].str.split("/")
    df['plane'] = df['plane'].str.split("/")

    return df

    

#this is where we will be adding the date to the dataFrame,
priceDf = getPriceListWithYear()



# otherDf = getPriceList()

# print('\n')
# print(priceDf)

# print('\n')
# print(otherDf)




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




def makeTableWithYear():
    #just a basic making of a iphone list

    print('\n\n\ninFunction\n\n\n')
    ipl = [x for x in priceDf['iphone']]# iPhone labels
    yr = [y for y in priceDf['year']]
    dy = [d for d in priceDf['year']]
    pc = [int(x[0]) for x in priceDf['price']] 




    #this is taking the values that were being inputed and turing them into lists

    #this is getting the value for the year
    for i in range(0, len(yr)):
        yr[i] = yr[i].year
        dy[i] = dy[i].day

    #this is needed to get the numbers or to turn it into numbers
    pc = np.array(pc, dtype=np.float64)

    print(pc)
    print(yr)
    print(ipl)

    difIphone = []
    difPrice = []
    difYear = []

    # for i in range(0,len(yr)):
    #     if((yr[i], dy[i]) != (yr[i -1], dy[i - 1])):
    #         if(ipl[i]!='iphone 12 Mini' or ipl[i]!='iPhone 12 Pro' or ipl[i]!='iphone 12 Pro Max' ):
    #             difIphone.append(ipl[i])
    #             difPrice.append(pc[i])
    #             difYear.append(yr[i])

    #just making the lists from the data
    for i in range(0,len(yr)):
        if((yr[i], dy[i]) == (yr[i -1], dy[i - 1])):
            print('--------Same----------')
            # print(dy[i])
            # print(yr[i])
            # print(ipl[i])
            # print('\n')
        else:
            if(ipl[i]=='iphone 12 Mini' or ipl[i]=='iPhone 12 Pro ' or ipl[i]=='iphone 12 Pro Max' ):
                print('--------Same----------')
                # print(dy[i])
                # print(yr[i])
                # print(ipl[i])
                # print('\n')
            else:

                difIphone.append(ipl[i])
                difPrice.append(pc[i])
                difYear.append(yr[i])
                # print(dy[i])
                # print(yr[i])
                # print(ipl[i])
                # print('\n')




    intrest = (difPrice[len(difPrice)-3] - difPrice[len(difPrice)-1]) / difPrice[len(difPrice)-1]

    # Initialize future values list
    futureValue = []

    # Calculate future values
    for i in range(1, 6):
        futureValue.append((difPrice[len(difPrice)-1]) + intrest)

    

    # Append future values to difPrice
    for value in futureValue:
        difPrice.append(value)

    # Extend difYear to include future years
    future_years = [difYear[-1] + i for i in range(1, 6)]
    difYear.extend(future_years)

    style.use('ggplot')

    # x-values for the existing and future data
    howManyPhones = np.arange(len(difIphone) + 5)

    # Calculate the line of best fit
    slope, intercept = np.polyfit(howManyPhones, difPrice, 1)

    # Calculate the best fit line
    best_fit_line = slope * howManyPhones + intercept

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(difYear, difPrice, marker='o', label='iPhone Prices')
    plt.plot(difYear, best_fit_line, color='green', linestyle='--', label='Best Fit Line')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.title('iPhone Prices by Year')
    plt.xticks(difYear, rotation=45)
    plt.yticks(difPrice, rotation=45)
    plt.legend()

    # Annotate the points with iPhone prices
    for i, iphone in enumerate(difIphone):
        plt.annotate(f'{iphone}', (difYear[i], difPrice[i]), textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    # Annotate the future points
    for i, year in enumerate(future_years):
        plt.annotate(f'Predicted', (year, futureValue[i]), textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()






    '''


    futureValue = []

    #finding the intrest amount between iphone 11, and 12
    intrest = (difPrice[len(difPrice)-1] - difPrice[len(difPrice)-3])/difPrice[len(difPrice)-1]


    futureValue.append((difPrice[len(difPrice)-1] * (1 + intrest))* 1)
    futureValue.append((difPrice[len(difPrice)-1] * (1 + intrest))* 2)
    futureValue.append((difPrice[len(difPrice)-1] * (1 + intrest))* 3)
    futureValue.append((difPrice[len(difPrice)-1] * (1 + intrest))* 4)
    futureValue.append((difPrice[len(difPrice)-1] * (1 + intrest))* 5)

    # print(futureValue)


    for i in range(0, len(futureValue)):
        difPrice.append(futureValue[i])
    
    style.use('ggplot')

    #the reasion 5 is there is to show five more years
    howManyPhones = np.arange(len(difIphone) + 5)


    # Calculate the line of best fit, so the way polyfit is working is that it take in two list numbers
    #it than does some math on the numbers inputed and get out just 2 values, the first value is the slope
    #the secoend vlaue is the intersept
    slope, intercept = np.polyfit(howManyPhones, difPrice, 1)

    #this is a list of all the values, its basicly doing the slope value * each item in the lsit + the intersept value
    best_fit_line = slope * howManyPhones + intercept

    # print(best_fit_line)

    # Create the plot
    plt.figure(figsize=(10, 6))
    #this is ploting the year by the price
    plt.scatter(difYear, difPrice, marker='o', label='iPhone Prices')
    #this 
    plt.plot(difYear, best_fit_line, color='green', linestyle='--', label='Best Fit Line')

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Price (USD)')
    plt.title('iPhone Prices by year')
    plt.xticks(difYear,rotation=45)
    plt.legend()

    #this is adding the iphone name to each point
    for i, iphone in enumerate(difIphone):
        plt.annotate(f'{iphone}', (difYear[i], difPrice[i]), textcoords="offset points", xytext=(0,10), ha='center', rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

    '''

    '''
    #working
    slope, intercept = np.polyfit(pc, pc, 1)

    #this is a list of all the values, its basicly doing the slope value * each item in the lsit + the intersept value
    best_fit_line = slope * pc + intercept

    # print(best_fit_line)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(yr, pc, marker='o', label='iPhone Prices')
    plt.plot(yr, best_fit_line, color='green', linestyle='--', label='Best Fit Line')

    # Add labels and title
    plt.xlabel('iPhone Models')
    plt.ylabel('Price (USD)')
    plt.title('iPhone Prices by Model')
    plt.xticks(yr, rotation=90)
    plt.legend()

    #this is adding the numbers to each point
    for i, price in enumerate(ipl):
        plt.annotate(f'{price}', (yr[i], pc[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Show the plot
    plt.tight_layout()
    plt.show()

    '''



    # print(yr)
    # print('\n')

    # for i in range(0,len(yr)):
    #     if(yr[i] == yr[i -1]):
    #         print('--------Same----------')
    #         print(yr[i])
    #         print(iphone[i])
            
    #         print('\n')
    #     else:
    #         print(yr[i])
    #         print(iphone[i])
    #         print('\n')
    #         newIphone.append(iphone[i])
    #         newYear.append(yr[i])
    #         newPrice.append(pc[i])

    # print(yr[1])
    # print('\n')
    # print(iphone)
    # print(len(newIphone))
    # print(len(newYear))



    # plt.plot(newYear,newPrice)

    # plt.scatter(newYear, newPrice, color='red', label='Scatter Plot')


    # plt.xticks(newYear, rotation=90)
    # plt.yticks(newPrice)


    # for i, iphone in enumerate(newIphone):
    #     plt.annotate(f'{iphone}', (newYear[i], newPrice[i]), textcoords="offset points", xytext=(0,10), ha='center')



    # # Add labels and title
    # plt.xlabel('Year')
    # plt.ylabel('Price')
    # plt.title('Year vs Price')

    # # Add a legend
    # plt.legend()



    # plt.show()

    #this is getting the future value, 
    # i = 0
    # #FV=PV(1+i)n
    #future value = present value x (1 + interest rate)n.
    #Interest Rate = (Simple Interest × 100)/(Principal × Time).

    #it will looks something like
    #plotValue = (Present value * (1 + intrest rate)) * year


    # style.use('ggplot')
    # plt.figure(figsize=(10, 10))
    # plt.plot(yr, pc, color='green', linestyle='--', label='Best Fit Line')


    # plt.show()


    #(final - init)
    #---------------        * 100  == intrest Rate
    #init*Time


    # intrest = (((fv.iloc[len(xs)-1]   -  fv.iloc[len(xs)-2])   /    (fv.iloc[len(xs)-1]*year)) * 100)
    # print(intrest)

    # next_value = (fv.iloc[len(xs)-1] * (1 + intrest)   )*year
    # print(fv.iloc[len(xs)-1])
    # print(next_value)




    

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


    '''
    # Plotting
    style.use('ggplot')

    #so this is the x values, its the lengh of the ipones[0,1,2,3,4,5,6,7,8,9]
    x_values = np.arange(len(iphone))
    #x_values = np.arange(len(iphone) + year)#this is for each iphone plus the number of years you want

    # Calculate the line of best fit, so the way polyfit is working is that it take in two list numbers
    #it than does some math on the numbers inputed and get out just 2 values, the first value is the slope
    #the secoend vlaue is the intersept
    slope, intercept = np.polyfit(x_values, ys, 1)

    #this is a list of all the values, its basicly doing the slope value * each item in the lsit + the intersept value

    #this is just a list of all the points 
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
    '''
    #Ending of the function




#this is a working version of the table This is the first Graph
y = [int(x[0]) for x in priceDf['price']]  # First price list
x = [int(x[0]) for x in priceDf['price']]  # first price list
makeBasicTable(x,y)



#this is what is running for the table
makeTableWithYear()






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




