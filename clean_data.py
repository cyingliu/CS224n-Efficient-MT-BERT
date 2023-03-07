import pandas as pd

# read the input file using pandas
df = pd.read_csv('SICK_train.txt', sep='	', header=0)

# drop the first and last columns
df = df.iloc[:, :-1]

df.columns = ['id', 'sentence1', 'sentence2', 'similarity']
              
# write the modified dataframe to a new text file
df.to_csv('SICK_train_cleaned.txt', sep='	', index=False)
