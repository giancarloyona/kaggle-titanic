"""
    Kaggle Getting Started Prediction Competition

    This is a starter's attempt to generate a survival predictor using 
    Titanic's sinking data.
    
    This model will use the Decision Tree Regression as the prediction
    algorithm.
    
    @author: giancarloyona
    @email: gianyona[at]gmail.com
    @date: jan/19

    TO DO: perform the preprocessing steps on the test dataset

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
from sklearn.impute import SimpleImputer

imputer_training = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_training  = imputer_training.fit(training_set[:, [0,1,3,4,5]])
training_set[:, [0,1,3,4,5]] = imputer_training.transform(training_set[:, [0,1,3,4,5]])

## encoding the variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

label_encoder = LabelEncoder()
training_set[:, 2] = label_encoder.fit_transform(training_set[:, 2])
training_set[:, 6] = label_encoder.fit_transform(training_set[:, 6])

column_transform = ColumnTransformer([
        ('Pclass', OneHotEncoder(categories='auto'), [1]),
        ('Sex', OneHotEncoder(categories='auto'), [2]),
        ('Embarked', OneHotEncoder(categories='auto'), [6])
], remainder='passthrough')

training_set = column_transform.fit_transform(training_set)
training_set = training_set.astype(np.float64)

# removing  dummy variables to avoid the dummy variable trap 

# generating the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'mse', 
                                  splitter = 'best', 
                                  random_state = 0)

# fitting the model
#regressor.fit(training_set, test_set)
