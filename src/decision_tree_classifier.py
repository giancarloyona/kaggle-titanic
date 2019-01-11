"""
    Kaggle Getting Started Prediction Competition

    This is a starter's attempt to generate a survival predictor using 
    Titanic's sinking data.
    
    This is a revamped version of my previous model 
    Based on the work of Mr. Ahmed (found at https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html)
    
    @author: giancarloyona
    @email: gianyona[at]gmail.com
    @date: jan/19
    @version: 0.1.1
"""

# importing the datasets
import pandas as pd
import numpy as np

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# preparing the data
target = train_dataset[['Survived']]
train_dataset.drop(labels = ['Survived'], inplace = True, axis = 1)

labels = train_dataset.columns.values
full_dataset = pd.DataFrame(np.append(train_dataset, test_dataset, axis = 0), columns = labels)
full_dataset.drop(labels = ['PassengerId'], inplace = True, axis = 1)

del labels

"""
# exploring the datasets

train_dataset.head()
test_dataset.head()

train_dataset.describe()
test_dataset.describe()
"""

# modelling the data
title = set()

for name in full_dataset['Name']:
    title.add(name.split(',')[1].split('.')[0].strip())

title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona" : "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

full_dataset['Title'] = full_dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
full_dataset['Title'] = full_dataset['Title'].map(title_dict)
full_dataset.drop(labels = ['Name'], inplace = True, axis = 1)

# imputing missing data
from sklearn.impute import SimpleImputer

imputer_age = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer_age = imputer_age.fit(np.reshape(full_dataset.iloc[:, 2].values, (-1, 1)))
full_dataset['Age'] = imputer_age.transform(np.reshape(full_dataset.iloc[:, 2].values, (-1, 1)))

imputer_fare = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer_fare = imputer_fare.fit(np.reshape(full_dataset.iloc[:, 6].values, (-1, 1)))
full_dataset['Fare'] = imputer_fare.transform(np.reshape(full_dataset.iloc[:, 6].values, (-1, 1)))

imputer_embarked = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer_embarked = imputer_embarked.fit(np.reshape(full_dataset.iloc[:, 8].values, (-1, 1)))
full_dataset['Embarked'] = imputer_embarked.transform(np.reshape(full_dataset.iloc[:, 8].values, (-1, 1)))

# encoding the categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

label_encoder = LabelEncoder()
full_dataset['Pclass'] = label_encoder.fit_transform(full_dataset['Pclass'])
full_dataset['Sex'] = label_encoder.fit_transform(full_dataset['Sex'])
full_dataset['Embarked'] = label_encoder.fit_transform(full_dataset['Embarked'])
full_dataset['Title'] = label_encoder.fit_transform(full_dataset['Title'])
