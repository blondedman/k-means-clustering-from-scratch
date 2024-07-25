from sklearn import preprocessing, model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np

style.use('ggplot')

df = pd.read_excel('titanic.xls')
print(df.head())

df.drop(['body','name'], axis=1, inplace=True)

# this does not work
# df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

columns = df.columns.tolist()

def converting(df):
    enc = LabelEncoder()
    
    enc.fit(df['sex'])
    df['sex'] = enc.transform(df['sex'])
    
    enc.fit(df['cabin'])
    df['cabin'] = enc.transform(df['cabin'])
    
    enc.fit(df['embarked'])
    df['embarked'] = enc.transform(df['embarked'])
    
    enc.fit(df['home.dest'])
    df['home.dest'] = enc.transform(df['home.dest'])
    
    return df
    
df = converting(df)
# print(df.head())

# one data type at a time
# while using encoders
df.fillna(0, inplace=True)

# can use something called "one-hot" encoding
# this will split the categorical feature into multiple features of all the different options 

# for a male entry, the features will be:
# sex_male = 1, sex_female = 0

# for a female entry, the features will be:
# sex_male = 0, sex_female = 1

def handling(df):
    columns = df.columns.values
    
    for column in columns:
        
        text_digit_vals = {}
        
        def convert_to_int(val):
            return text_digit_vals [val]
        
        if df [column].dtype != np.int64 and df [column].dtype != np.float64: 
            column_contents = df [column].values.tolist()                
            unique_elements = set (column_contents)
            x = 0
                
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df [column] = list(map(convert_to_int, df[column]))
            
    return df

df = handling(df)
# print(df.head())

# dropping various columns
# to increase accuracy
# df.drop([''], axis=1)

X = np.array(df.drop(['survived'], axis=1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1,len(predict))
    
    prediction = clf.predict(predict)
    
    if prediction[0] == y[i]:
        correct += 1 
        
print(correct/len(X))