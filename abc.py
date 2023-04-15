

import numpy as np
import pandas as pd
from sklearn import preprocessing
  
# Import dataset
df = pd.read_csv('train.csv')
  

label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['DayOfWeek']= label_encoder.fit_transform(df['DayOfWeek'])
  
df['DayOfWeek'].unique()