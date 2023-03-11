import json
import random
"""
# Define the filename
filename = 'reviews_Kindle_Store_5.json'

# Open the JSON file for reading
with open(filename, 'r') as json_file:

    # Load JSON data
    data = [json.loads(line) for line in json_file]

    # Define headers to keep
    headers_to_keep = ['reviewerID', 'reviewText', 'overall']

    # Create new list of dictionaries with only desired headers
    new_data = [{h: row[h] for h in headers_to_keep} for row in data]

# Open a new file for writing
with open('new_data.json', 'w') as json_file:

    # Write the modified data to the new file
    for row in new_data:
        json.dump(row, json_file)
        json_file.write('\n')
"""

# Open the JSON file for reading
with open('new_data.json', 'r') as json_file:

    # Load JSON data
    data = [json.loads(line) for line in json_file]

    # Get the headers
    headers = list(data[0].keys())

# Open a new file for writing
with open('kindle_data_train.txt', 'w') as txt_file:

    # Write the headers to the new file
    txt_file.write('\t'.join(headers) + '\n')

    # Write the data to the new file
    for row in data:
        values = [str(row[h]).replace('\t', ' ') for h in headers]
        txt_file.write('\t'.join(values) + '\n')

# Open the input file for reading
with open('kindle_data_train.txt', 'r') as input_file:

    # Read the header line and get the headers
    header_line = input_file.readline()
    headers = header_line.strip().split('\t')

    # Read the rest of the lines and store in a list
    lines = input_file.readlines()

# Randomly sample 100,000 lines from the list
sampled_lines = random.sample(lines, k=100000)

# Open the output file for writing
with open('kindle_data_train_cleaned.txt', 'w') as output_file:

    # Write the header line to the output file
    output_file.write(header_line)

    # Write the sampled lines to the output file
    for line in sampled_lines:
        output_file.write(line)
"""
import pandas as pd

# read the input file using pandas
df = pd.read_csv('SICK_train.txt', sep='	', header=0)

#df.insert(0, 'pair_ID_duplicate', df['pair_ID'])

# drop the last columns
df = df.iloc[:, :-1]

df.columns = ['id', 'sentence1', 'sentence2', 'similarity']
              
# write the modified dataframe to a new text file
df.to_csv('SICK_train_cleaned.txt', sep='	', index=True)
"""