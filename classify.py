import pandas as pd


coms = pd.read_csv('~/Desktop/text-model/Coms.csv')

def clean_text(df, column):
    coms['Text_clean'] = coms['Comment'].str.replace(r'[^A-Za-z0-9]','')
    coms['Text_clean'] = coms['Comment'].str.lower()
    return df

coms_cl = clean_text(coms, 'Comment')
# print(coms['Labels'].value_counts())
print(coms_cl.head())
