"""
    Kaggle Getting Started Prediction Competition

    This is a starter's attempt to generate a survival predictor using 
    Titanic's sinking data.
    
    This first model will use the Decision Tree Regression as the prediction
    algorithm.
    
    @author: giancarloyona
    @email: gianyona[at]gmail.com
    @date: jan/19
    @version: 0.1
"""

# importing the dataset
import numpy as np
import pandas as pd

training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

# preprocessing the data
X_train = training_set.iloc[:, [2, 4, 5, 6, 7, 11]].values
Y_train = training_set.iloc[: , 1].values
Y_train = np.reshape(Y_train, (-1, 1))

test_set = test_set.iloc[:, [1, 3, 4, 5, 6, 10]].values

## replacing missing values

### replacing null values
X_train[pd.isnull(X_train)] = 'NaN'

### replacing numeric values
from sklearn.impute import SimpleImputer

imputer_training = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_training  = imputer_training.fit(X_train[:, [0, 2, 3, 4]])
X_train[:, [0, 2, 3, 4]] = imputer_training.transform(X_train[:, [0, 2, 3, 4]])

imputer_training_1 = SimpleImputer(missing_values='NaN', strategy='most_frequent')
imputer_training_1 = imputer_training_1.fit(X_train[:, [1, 5]])
X_train[: , [1, 5]] = imputer_training_1.transform(X_train[:, [1, 5]])

imputer_test = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_test = imputer_test.fit(test_set[: , [0, 2, 3, 4]])
test_set[: , [0, 2, 3, 4]] = imputer_test.transform(test_set[: , [0, 2, 3, 4]])

imputer_test_1 = SimpleImputer(missing_values='NaN', strategy='most_frequent')
imputer_test_1 = imputer_test_1.fit(test_set[:, [1, 5]])
test_set[: , [1, 5]] = imputer_test_1.transform(test_set[:, [1, 5]])

## encoding the variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

### encoding the train data
label_encoder_train = LabelEncoder()
X_train[:, 0] = label_encoder_train.fit_transform(X_train[:, 0])
X_train[:, 1] = label_encoder_train.fit_transform(X_train[:, 1])
X_train[:, 5] = label_encoder_train.fit_transform(X_train[:, 5])

column_transform_train = ColumnTransformer(
        [('Pclass', OneHotEncoder(categories='auto'), [0]),
         ('Sex', OneHotEncoder(categories='auto'), [1]),
         ('Embarked', OneHotEncoder(categories='auto'), [5])
], remainder='passthrough')

X_train = column_transform_train.fit_transform(X_train)
X_train = X_train.astype(np.float64)

### encoding the test data
label_encoder_test = LabelEncoder()
test_set[:, 0] = label_encoder_test.fit_transform(test_set[:, 0])
test_set[:, 1] = label_encoder_test.fit_transform(test_set[:, 1])
test_set[:, 5] = label_encoder_test.fit_transform(test_set[:, 5])

column_transform_train = ColumnTransformer(
        [('Pclass', OneHotEncoder(categories='auto'), [0]),
         ('Sex', OneHotEncoder(categories='auto'), [1]),
         ('Embarked', OneHotEncoder(categories='auto'), [5])
], remainder='passthrough')

test_set = column_transform_train.fit_transform(test_set)
test_set = test_set.astype(np.float64)

# removing  dummy variables to avoid the dummy variable trap 

# generating the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = 'mse', 
                                  splitter = 'best', 
                                  random_state = 0)

# fitting the model
regressor.fit(X_train, Y_train)

# predicting the results
prediction = regressor.predict(test_set)
prediction = np.reshape(prediction, (-1, 1))
np.around(prediction, 0)
