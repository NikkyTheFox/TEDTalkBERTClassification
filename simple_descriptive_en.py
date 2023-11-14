import pandas as pd
import random
df = pd.read_csv('ted_talks_en.csv')

id_column = df['talk_id']
topics_column = df['topics']
description_column = df['description']
transcript_column = df['transcript']

# topics_average_length = topics_column.apply(lambda x: len(eval(x))).mean()
# topics_std_length = topics_column.apply(lambda x: len(eval(x))).std()

# print(topics_average_length)
# print(topics_std_length)

def split_text(text):
    return text.strip("[]''").replace("', '", ',').split(', ')

def extract_word(text):
    arr = text[0].split(',')
    if len(arr) > 0:
        return arr[random.randint(0, len(arr) - 1)]
    else:
        return None

df2 = pd.DataFrame({
    'id': id_column, 
    'topics': topics_column, 
    'description': description_column, 
    'transcript': transcript_column})

print(df2.dtypes)

df2['topics'] = df2['topics'].astype(str).apply(split_text)
df2['topic'] = df2['topics'].apply(extract_word)

df2.to_csv('test.csv', index=False)

df2 = pd.read_csv('test.csv')

print(df2.dtypes)
print(df2.head())
print(df2['topic'].nunique())
print(df2.count())