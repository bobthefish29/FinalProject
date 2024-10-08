
#Ive used some of the ideas from, this version of the file is from when it was first added to the branch,


import pandas as pd

# Load the DataFrame skip extra header rows if neccessary
df = pd.read_csv(r'data/SeedUnofficialAppleDataCSV.csv', encoding='latin1', skiprows=2)

# Rename columns
df.columns = ['model', 'release_os', 'release_date', 'discontinued', 'support_ended',
            'final_os', 'lifespan', 'max_lifespan', 'launch_price']

# Remove any rows where 'model' is NaN
df = df[df['model'].notna()]

# Convert date columns to datetime objects with error handling
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce', format="MM/DD/YYYY")
df['discontinued'] = pd.to_datetime(df['discontinued'], errors='coerce', format="MM/DD/YYYY")
df['support_ended'] = pd.to_datetime(df['support_ended'], errors='coerce', format="MM/DD/YYYY")

print(df.columns)


print(df.head(10))





