from sklearn import preprocessing, model_selection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np

style.use('ggplot')

df = pd.read_excel('titanic.xls')
print(df.head())