import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing libraries:
# 1. pandas: Used for data manipulation of the CSV file.
# 2. matplotlib.pyplot: used to create plots and graphs.
# 3. numpy: not used in this graph but something we have worked with and is usually included for numerical operations if needed.

# Reading CSV file and assigning column names
file = pd.read_csv(r'data/SeedUnofficialAppleDataCSV.csv')
file.columns = ['model', 'release_os', 'release_date', 'discontinued', 'support_ended', 'final_os', 'lifespan', 'max_lifespan', 'launch_price']


# function to get a list of all the iPhones and their prices
def getPhoneList():
    # drop the rows where model is NaN, only keeping valid model entries
    tfile = file.dropna(subset='model', inplace=False, axis='index')
    phoneList = []  # list to store each phone model as a sublist split by '/'
    phoneList2 = []  # list to store individual phone models after the sublist split

    # loop over each iPhone model entry in the model column and split by '/' (showing different version of the model).
    for i in tfile['model']:    
        phoneList.append(i.split("/"))

    # loop through the phoneList and further split any items with multiple versions.
    for i in range(1, len(phoneList)):
        if len(phoneList[i]) > 1:
            for l in range(0, len(phoneList[i])):
                phoneList2.append(phoneList[i][l])
        else:
            phoneList2.append(phoneList[i][0])

    # fix any models that start with a space by removing the space and adding the word iphone to the beginning.
    for i in range(1, len(phoneList2)):
        if phoneList2[i][0] == " ":
            phoneList2[i] = "iphone" + phoneList2[i]
    
    # return the final cleaned list of phone models.
    return phoneList2

# Function to get the price list and clean the data
def getPriceList():
    tempFile = file  # use a temporary copy of the file to work with
    result_list = []  # store models and their associated prices
    realFinal = []  # store cleaned and structured data

    # loop over each row in the CSV file - not sure if this is doing what we want it to do
    for i, r in tempFile.iterrows():
        if pd.isnull(r['model']):  # if the model column is NaN, assume it's a continuation of the previous model.
            result_list[-1][1].append(r['launch_price'])  # append the price to the last model's price list.
        else:
            result_list.append([r['model'], [r['launch_price']]])  # else create a new entry with the model and price.

    # clean the models and prices in the result list.
    for i in range(1, len(result_list)):
        result_list[i][0] = result_list[i][0].split("/")  # splitting the models that have multiple versions.

        # fixing any model names that start with a space.
        for l in range(0, len(result_list[i][0])):
            if result_list[i][0][l].startswith(" "):
                result_list[i][0][l] = "iphone" + result_list[i][0][l]

        # clean the price data by removing unwanted characters (*, $, Plus:, Max:, Mini:).
        for l in range(0, len(result_list[i][1])):
            if isinstance(result_list[i][1][l], str):  # Ensure the price is a string before attempting to clean it.
                result_list[i][1][l] = result_list[i][1][l].replace('*', '').replace('$', '').replace('Plus:', '').replace('Max:', '').replace('Mini:', '')

    # formatting data to more clean format
    for i in range(1, len(result_list)):
        if len(result_list[i][0]) == 1:  # If there's only one model.
            if len(result_list[i][1]) > 1:  # If there are multiple prices, structure it accordingly.
                realFinal.append([result_list[i][0][0], result_list[i][1][0], result_list[i][1][1]])
            else:
                realFinal.append([result_list[i][0][0], result_list[i][1][0]])
        else:  # If there are multiple models.
            if len(result_list[i][1]) > 2:  # Handle cases with more than 2 prices.
                for l in range(0, len(result_list[i][0])):
                    if l == 0:
                        realFinal.append([result_list[i][0][l], result_list[i][1][0], result_list[i][1][1]])
                    else:
                        realFinal.append([result_list[i][0][l], result_list[i][1][2], result_list[i][1][3]])
            else:  # Handle cases with 2 prices.
                for l in range(0, len(result_list[i][0])):
                    if l == 0:
                        realFinal.append([result_list[i][0][l], result_list[i][1][0]])
                    else:
                        realFinal.append([result_list[i][0][l], result_list[i][1][1]])

    # Creating a DataFrame from the final cleaned list of models and prices.
    df = pd.DataFrame(realFinal, columns=['iphone', 'price', 'plane'])

    # Convert 'price' and 'plane' fields into lists if they contain multiple values (split by "/").
    df['price'] = df['price'].apply(lambda x: x.split("/") if isinstance(x, str) else [x])
    df['plane'] = df['plane'].apply(lambda x: x.split("/") if isinstance(x, str) else [x])

    # Return the cleaned DataFrame.
    return df

# Fetch the cleaned-up iPhone data and prices
priceDf = getPriceList()

# Filter out rows where 'price' is NaN or invalid (i.e., rows with missing or corrupt price data).
priceDf = priceDf.dropna(subset=['price'])

# Flatten the 'iphone' and 'price' data into simple lists to use for plotting.
x = [item for item in priceDf['iphone']]  # List of iPhone models.
y = [int(val[0]) if isinstance(val, list) else int(val) for val in priceDf['price']]  # List of iPhone prices (converted to integers).

# Ensure the x and y lists (models and prices) have the same length.
if len(x) != len(y):
    print("Error: Length mismatch between iPhone models and prices.")  # Print error if they don't match.
    exit(1)  # Exit the script if there's an error.

# Create a figure and axes for plotting.
fig, ax = plt.subplots()

# Plot the iPhone models (x) against their prices (y).
ax.plot(x[:len(y)], y, label="iPhone Prices", marker="o")

# Add title and labels to the plot.
ax.set_title("iPhone Prices Over Time")
ax.set_xlabel("iPhone Model")
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability.
ax.set_ylabel("Price")

# Add a legend to the plot.
ax.legend()

# Display the plot.
plt.show()
