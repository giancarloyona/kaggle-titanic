"""
    Kaggle Getting Started Prediction Competition

    This is an attempt to generate a survival predictor using Titanic's sinking
    data.
    
    @author: giancarloyona
    @email: gianyona[at]gmail.com
    @date: jan/19
"""

# importing the dataset

import numpy as np
import pandas as pd

training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# preprocessing the data
training_set = training_set.iloc[:, [1,2,4,5,6,7,11]].values
test_set = test_set.iloc[:, [1,3,4,5,6,10]]

## replacing missing values

### replacing NaN to string values
training_set[pd.isnull(training_set)] = 'NaN'

### replacing numeric values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
training_set[:, [0,1,3,4,5]] = imputer.fit_transform(training_set[:, [0,1,3,4,5]])

## encoding the variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# generating the model

# fitting the model

# visualizating the results