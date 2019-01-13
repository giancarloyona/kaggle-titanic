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

# extracting the title of each passenger
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

del name

full_dataset['Title'] = full_dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
full_dataset['Title'] = full_dataset['Title'].map(title_dict)
full_dataset.drop(labels = ['Name'], inplace = True, axis = 1)

# parsing the ticket data

full_dataset['Ticket'] = full_dataset['Ticket'].map(lambda ticket:ticket.replace('/', '').replace('.', '').split()[0])
full_dataset['Ticket'] = full_dataset['Ticket'].map(lambda ticket:'U' if ticket.isdigit() else ticket[0])

# imputing missing data
from sklearn.impute import SimpleImputer

imputer_age = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer_age = imputer_age.fit(np.reshape(full_dataset.iloc[:, 2].values, (-1, 1)))
full_dataset['Age'] = imputer_age.transform(np.reshape(full_dataset.iloc[:, 2].values, (-1, 1)))

imputer_fare = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer_fare = imputer_fare.fit(np.reshape(full_dataset.iloc[:, 6].values, (-1, 1)))
full_dataset['Fare'] = imputer_fare.transform(np.reshape(full_dataset.iloc[:, 6].values, (-1, 1)))

imputer_cabin = SimpleImputer(missing_values = np.nan, fill_value = "U", strategy = 'constant')
imputer_cabin = imputer_cabin.fit(np.reshape(full_dataset.iloc[:, 7].values, (-1, 1)))
full_dataset['Cabin'] = imputer_cabin.transform(np.reshape(full_dataset.iloc[:, 7].values, (-1, 1)))
full_dataset['Cabin'] = full_dataset['Cabin'].map(lambda c: c[0])

imputer_embarked = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer_embarked = imputer_embarked.fit(np.reshape(full_dataset.iloc[:, 8].values, (-1, 1)))
full_dataset['Embarked'] = imputer_embarked.transform(np.reshape(full_dataset.iloc[:, 8].values, (-1, 1)))
   
# encoding the categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

label_encoder = LabelEncoder()
full_dataset['Pclass'] = label_encoder.fit_transform(full_dataset['Pclass'])
full_dataset['Sex'] = label_encoder.fit_transform(full_dataset['Sex'])
full_dataset['Ticket'] = label_encoder.fit_transform(full_dataset['Ticket'])
full_dataset['Cabin'] = label_encoder.fit_transform(full_dataset['Cabin'])
full_dataset['Embarked'] = label_encoder.fit_transform(full_dataset['Embarked'])
full_dataset['Title'] = label_encoder.fit_transform(full_dataset['Title'])

column_tranformer = ColumnTransformer(
        [('Pclass', OneHotEncoder(categories = 'auto'), [0]),
         ('Sex', OneHotEncoder(categories = 'auto'), [1]),
         ('Ticket', OneHotEncoder(categories = 'auto'), [5]),
         ('Cabin', OneHotEncoder(categories = 'auto'), [7]),
         ('Embarked', OneHotEncoder(categories = 'auto'), [8]),
         ('Title', OneHotEncoder(categories = 'auto'), [9])
], remainder='passthrough', sparse_threshold=0)

full_dataset = column_tranformer.fit_transform(full_dataset)

# removing a few dummy variables
full_dataset = pd.DataFrame(full_dataset)
full_dataset = full_dataset.drop(full_dataset.columns[[0, 3, 5, 14, 17]], axis = 1)

# generating the model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, Y_train, Y_test = train_test_split(full_dataset[:891], target, test_size = 0.2, random_state = 42)

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
Y_train = Y_train.astype(np.float64)
Y_test = Y_test.astype(np.float64)

regressor = DecisionTreeClassifier(criterion = 'gini', splitter='best', random_state = 42)
regressor.fit(X_train, Y_train)
prediction = regressor.predict(X_test)

# assessing the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, prediction)