from sklearn import preprocessing, model_selection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np

style.use('ggplot')

df = pd.read_excel('titanic.xls')
# print(df.head())

df.drop(['body','name'], axis=1, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
print(df.head())