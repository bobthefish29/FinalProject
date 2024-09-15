import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading CSV file
file = pd.read_csv(r'data/SeedUnofficialAppleDataCSV.csv')
file.columns = ['model', 'release_os', 'release_date', 'discontinued', 'support_ended', 'final_os', 'lifespan', 'max_lifespan', 'launch_price']

# Function to get a list of all the iPhones and their prices
def getPhoneList():
    tfile = file.dropna(subset='model', inplace=False, axis='index')
    phoneList = []
    phoneList2 = []
    for i in tfile['model']:    
        phoneList.append(i.split("/"))

    for i in range(1, len(phoneList)):
        if len(phoneList[i]) > 1:
            for l in range(0, len(phoneList[i])):
                phoneList2.append(phoneList[i][l])
        else:
            phoneList2.append(phoneList[i][0])

    for i in range(1, len(phoneList2)):
        if phoneList2[i][0] == " ":
            phoneList2[i] = "iphone" + phoneList2[i]
    return phoneList2

# Function to get the price list and clean the data
def getPriceList():
    tempFile = file
    result_list = []
    realFinal = []

    for i, r in tempFile.iterrows():
        if pd.isnull(r['model']):
            result_list[-1][1].append(r['launch_price'])
        else:
            result_list.append([r['model'], [r['launch_price']]])

    for i in range(1, len(result_list)):
        result_list[i][0] = result_list[i][0].split("/")

        for l in range(0, len(result_list[i][0])):
            if result_list[i][0][l].startswith(" "):
                result_list[i][0][l] = "iphone" + result_list[i][0][l]

        for l in range(0, len(result_list[i][1])):
            if isinstance(result_list[i][1][l], str):
                result_list[i][1][l] = result_list[i][1][l].replace('*', '').replace('$', '').replace('Plus:', '').replace('Max:', '').replace('Mini:', '')

    for i in range(1, len(result_list)):
        if len(result_list[i][0]) == 1:
            if len(result_list[i][1]) > 1:
                realFinal.append([result_list[i][0][0], result_list[i][1][0], result_list[i][1][1]])
            else:
                realFinal.append([result_list[i][0][0], result_list[i][1][0]])
        else:
            if len(result_list[i][1]) > 2:
                for l in range(0, len(result_list[i][0])):
                    if l == 0:
                        realFinal.append([result_list[i][0][l], result_list[i][1][0], result_list[i][1][1]])
                    else:
                        realFinal.append([result_list[i][0][l], result_list[i][1][2], result_list[i][1][3]])
            else:
                for l in range(0, len(result_list[i][0])):
                    if l == 0:
                        realFinal.append([result_list[i][0][l], result_list[i][1][0]])
                    else:
                        realFinal.append([result_list[i][0][l], result_list[i][1][1]])

    df = pd.DataFrame(realFinal, columns=['iphone', 'price', 'plane'])

    # Converting price and plane to lists, flattening them
    df['price'] = df['price'].apply(lambda x: x.split("/") if isinstance(x, str) else [x])
    df['plane'] = df['plane'].apply(lambda x: x.split("/") if isinstance(x, str) else [x])

    return df

# Fetch the cleaned-up iPhone data and prices
priceDf = getPriceList()

# Filter out rows where 'price' is NaN or invalid
priceDf = priceDf.dropna(subset=['price'])

# Function to calculate future prices
def calculate_future_price(current_price, growth_rate, years):
    return current_price * (1 + growth_rate) ** years

# Assume a growth rate of 5% as a best practice - we would need to 
growth_rate = 0.05

# Calculate future prices for each iPhone model
priceDf['price'] = priceDf['price'].apply(lambda x: [val for val in x if val.isdigit()])  # Filter out non-digit values

# Convert the list of prices to a single float value
priceDf['price'] = priceDf['price'].apply(lambda x: float(x[0]) if len(x) > 0 else 0)

# Prepare data for plotting
x = [item for item in priceDf['iphone']]
y = [val for val in priceDf['price']]

# Calculate future prices
future_prices_2_years = [calculate_future_price(price, growth_rate, 2) for price in y]
future_prices_3_years = [calculate_future_price(price, growth_rate, 3) for price in y]
future_prices_5_years = [calculate_future_price(price, growth_rate, 5) for price in y]

# Create a figure and axes
fig, ax = plt.subplots()

# Plot historical prices
ax.plot(x, y, label="Historical iPhone Prices", marker="o", color='blue')

# Plot future prices
ax.plot(x, future_prices_2_years, label="Projected Prices (2 years)", linestyle='--', color='green')
ax.plot(x, future_prices_3_years, label="Projected Prices (3 years)", linestyle='--', color='orange')
ax.plot(x, future_prices_5_years, label="Projected Prices (5 years)", linestyle='--', color='red')

# Add a title and labels
ax.set_title("iPhone Prices Over Time and Future Projections")
ax.set_xlabel("iPhone Model")
plt.xticks(rotation=90)
ax.set_ylabel("Price")

# Add a legend
ax.legend()

# Show the plot
plt.show()

#------------- Predicting the price for a specific model and year-------------------------
# After hitting x on the graph you can input the model and year in the terminal for a prediction
model_name = input("Enter the iPhone model name: ")
future_year = int(input("Enter the future year: "))

# Check if the model is in the DataFrame
if model_name in priceDf['iphone'].values:
    # Find the current price for the model
    current_price = priceDf.loc[priceDf['iphone'] == model_name, 'price'].values[0]
    
    # Calculate the number of years from now
    current_year = 2024  
    years_to_predict = future_year - current_year
    
    # Calculate the predicted price
    predicted_price = calculate_future_price(current_price, growth_rate, years_to_predict)
    
    # Print the predicted price
    print(f"Predicted price for {model_name} in {future_year}: ${predicted_price:.2f}")
else:
    print(f"{model_name} is not found in the data.")
