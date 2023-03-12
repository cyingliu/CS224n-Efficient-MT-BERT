import json
import random

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

MAX_WORDS = 30
MAX_ENTRIES = 15000
num_entries_with_rating = [0, 0, 0, 0, 0, 0]
with open("kindle_data_train.txt", "r", encoding="utf-8") as infile, \
     open("kindle_data_train_cleaned.txt", "w", encoding="utf-8") as outfile:

    # Write headers to output file
    headers = infile.readline().strip().split("\t")
    outfile.write("\t".join(headers) + "\n")

    # Read and filter entries
    entries = [line.strip().split("\t") for line in infile.readlines()]
    random.shuffle(entries)
    selected_entries = []
    for entry in entries:
        review_text = entry[1]
        num_words = len(review_text.split())
        if num_words <= MAX_WORDS and num_words > 1:
            rating = int(float(entry[2]))
            entry[2] = str(rating)
            if (num_entries_with_rating[rating] < 3000):
                selected_entries.append(entry)
                num_entries_with_rating[rating] += 1
            if len(selected_entries) >= MAX_ENTRIES:
                break

    # Write selected entries to output file
    for entry in selected_entries:
        outfile.write("\t".join(entry) + "\n")

with open('kindle_data_train_cleaned.txt', 'r') as f_in, open('kindle_data_train_fixed.txt', 'w') as f_out:
    headers = f_in.readline().strip().split('\t')
    f_out.write('\t'.join(headers) + '\n')
    for line in f_in:
        cols = line.strip().split('\t')
        cols[2] = str(int(cols[2]) - 1)
        f_out.write('\t'.join(cols) + '\n')


# fjoisdajf

import pandas as pd

# read the input file using pandas
df = pd.read_csv('kindle_data_train_fixed.txt', sep='	', header=0)


# drop the last columns
# df = df.iloc[:, :-1]

df.columns = ['id', 'sentence', 'similarity']
              
# write the modified dataframe to a new text file
df.to_csv('kindle_data_train_fixed.txt', sep='	', index=True)

