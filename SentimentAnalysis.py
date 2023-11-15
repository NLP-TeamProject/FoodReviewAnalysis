import pandas as pd
# Initialize an empty list to store the records
records = []

# file_path=
# Open and read the text file
with open(r'foods.txt', 'r', encoding='latin1') as file:
    record = {}  # Initialize an empty dictionary for each record
    for line in file:
        line = line.strip()
        if not line:  # Check for empty line to separate records
            if record:  # Append the record dictionary to the list
                records.append(record)
            record = {}  # Initialize a new record
        elif ':' in line:  # Check if the line contains a colon
            key, value = line.split(': ', 1)  # Split each line into key and value
            record[key] = value

# Append the last record (if it exists) since the file may not end with an empty line
if record:
    records.append(record)

# Create a DataFrame from the list of records
df = pd.DataFrame(records)

# Show the DataFrame.
df

"#########################"

#columns are renamed
df.columns=['productid','userid','name','helpfulness','score','time','summary','review']
df

df['HFN'] = df['helpfulness'].apply(lambda x: int(x.split('/')[0]))
df['HFD'] = df['helpfulness'].apply(lambda x: int(x.split('/')[1]))

df['Helpful %'] = np.where(df['HFD'] > 0, df['HFN'] / df['HFD'], -1)
df['% Upvote'] = pd.cut(df['Helpful %'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest=True)

df.isna().sum()
df.duplicated().sum()
df[df.duplicated()]
df = df.drop_duplicates(subset=['productid','userid','time','review'],keep='first',inplace=False)

"##########################""#########################"

#columns are renamed
df.columns=['productid','userid','name','helpfulness','score','time','summary','review']
df

df['HFN'] = df['helpfulness'].apply(lambda x: int(x.split('/')[0]))
df['HFD'] = df['helpfulness'].apply(lambda x: int(x.split('/')[1]))

df['Helpful %'] = np.where(df['HFD'] > 0, df['HFN'] / df['HFD'], -1)
df['% Upvote'] = pd.cut(df['Helpful %'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'], include_lowest=True)

df.isna().sum()
df.duplicated().sum()
df[df.duplicated()]
df = df.drop_duplicates(subset=['productid','userid','time','review'],keep='first',inplace=False)

"##########################"
