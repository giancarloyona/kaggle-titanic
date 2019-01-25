"""
    Kaggle Getting Started Prediction Competition

    This is a starter's attempt to generate a survival predictor using
    Titanic's sinking data.

    This is a revamped version of my previous model
    Based on the work of Mr. Ahmed
    Found at https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html

    This model will use the Gradient Boost Classifier to generate the results.

    __author__ = giancarloyona
    __email__ = gianyona@gmail.com
    __date__ = jan/19
    __version__ = 0.1.2.2
"""

# importing the libraries needed for the challenge
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

# importing the data sets
training_set = pd.read_csv("../datasets/train.csv")
test_set = pd.read_csv("../datasets/test.csv")

# preparing the data
target = training_set[['Survived']]
training_set.drop(labels=['Survived'], inplace=True, axis=1)

labels = training_set.columns.values
full_set = pd.DataFrame(np.append(training_set, test_set, axis=0), columns=labels)
full_set.drop(labels=['PassengerId'], inplace=True, axis=1)

# modelling the data
# extracting the title of each passenger

title = set()

for names in full_set['Name']:
    title.add(names.split(',')[1].split('.')[0].strip())

title_dict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

full_set['Title'] = full_set['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
full_set['Title'] = full_set['Title'].map(title_dict)
full_set.drop(labels=['Name'], inplace=True, axis=1)

# parsing the ticket data
full_set['Ticket'] = full_set['Ticket'].map(lambda ticket: ticket.replace('/', '').replace('.', '').split()[0])
full_set['Ticket'] = full_set['Ticket'].map(lambda ticket: 'U' if ticket.isdigit() else ticket[0])

# imputing missing data
imputer_age = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_age = imputer_age.fit(np.reshape(full_set.iloc[:, 2].values, (-1, 1)))
full_set['Age'] = imputer_age.transform(np.reshape(full_set.iloc[:, 2].values, (-1, 1)))

imputer_fare = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_fare = imputer_fare.fit(np.reshape(full_set.iloc[:, 6].values, (-1, 1)))
full_set['Fare'] = imputer_fare.transform(np.reshape(full_set.iloc[:, 6].values, (-1, 1)))

imputer_cabin = SimpleImputer(missing_values=np.nan, fill_value="U", strategy='constant')
imputer_cabin = imputer_cabin.fit(np.reshape(full_set.iloc[:, 7].values, (-1, 1)))
full_set['Cabin'] = imputer_cabin.transform(np.reshape(full_set.iloc[:, 7].values, (-1, 1)))
full_set['Cabin'] = full_set['Cabin'].map(lambda c: c[0])

imputer_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_embarked = imputer_embarked.fit(np.reshape(full_set.iloc[:, 8].values, (-1, 1)))
full_set['Embarked'] = imputer_embarked.transform(np.reshape(full_set.iloc[:, 8].values, (-1, 1)))

# encoding the categorical variables
label_encoder = LabelEncoder()
full_set['Pclass'] = label_encoder.fit_transform(full_set['Pclass'])
full_set['Sex'] = label_encoder.fit_transform(full_set['Sex'])
full_set['Ticket'] = label_encoder.fit_transform(full_set['Ticket'])
full_set['Cabin'] = label_encoder.fit_transform(full_set['Cabin'])
full_set['Embarked'] = label_encoder.fit_transform(full_set['Embarked'])
full_set['Title'] = label_encoder.fit_transform(full_set['Title'])

column_tranformer = ColumnTransformer(
    [('Pclass', OrdinalEncoder(categories='auto'), [0]),
     ('Sex', OneHotEncoder(categories='auto'), [1]),
     ('Ticket', OneHotEncoder(categories='auto'), [5]),
     ('Cabin', OneHotEncoder(categories='auto'), [7]),
     ('Embarked', OneHotEncoder(categories='auto'), [8]),
     ('Title', OneHotEncoder(categories='auto'), [9])
     ], remainder='passthrough', sparse_threshold=0)

full_set = column_tranformer.fit_transform(full_set)
full_set = pd.DataFrame(full_set)

# mapping the Pclass variable to an ordinal variable
# first class > second class > third class

pclass_dict = {
    2: 0,
    1: 1,
    0: 2
}

full_set[0] = full_set[0].map(pclass_dict)

del pclass_dict

# removing a few dummy variables
full_set = full_set.drop(full_set.columns[[1, 3, 10, 19, 22]], axis=1)

# generating the model
regressor = GradientBoostingClassifier(n_estimators=100, random_state=0)
regressor.fit(full_set[:891], np.ravel(target))
prediction = regressor.predict(full_set[891:])

# assessing the model
kfold = model_selection.KFold(n_splits=10, random_state=0)
k_fold_results = model_selection.cross_val_score(regressor, full_set[:891], np.ravel(target), cv=kfold)
print(k_fold_results.mean())

# exporting the results
index = np.reshape(np.arange(start=892, stop=1310), (-1, 1))
results = np.append(index, np.reshape(prediction, (-1, 1)), axis=1)
results = pd.DataFrame(results, columns=['PassengerId', 'Survived'])
results.to_csv('../datasets/results_gbc.csv', sep=',', index=False)
