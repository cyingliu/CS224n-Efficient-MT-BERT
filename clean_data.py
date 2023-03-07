import pandas as pd

# read the input file using pandas
df = pd.read_csv('SICK_train.txt', sep='	', header=0)

#df.insert(0, 'pair_ID_duplicate', df['pair_ID'])

# drop the last columns
df = df.iloc[:, :-1]

df.columns = ['id', 'sentence1', 'sentence2', 'similarity']
              
# write the modified dataframe to a new text file
df.to_csv('SICK_train_cleaned.txt', sep='	', index=True)
