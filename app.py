# import libraries
import pandas as pd
dataFile = "review.txt"
# read data from the text file
with open(dataFile, "r") as file:
    data = file.read()
    # split the data into individual reviews
reviews = data.strip().split("\n\n")
review_dicts = []
for review in reviews:
    review_dict = {}
    lines = review.split('\n')
    for line in lines:
        # Split each line at the first colon, assuming the field name may contain colons
        parts = line.split(':', 1)
        if len(parts) == 2:
            field_name, field_value = parts
            review_dict[field_name.strip()] = field_value.strip()
    review_dicts.append(review_dict)
    df = pd.DataFrame(review_dicts)
    print(df)