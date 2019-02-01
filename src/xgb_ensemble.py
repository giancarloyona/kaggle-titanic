"""
    Kaggle Getting Started Prediction Competition
    
    This is a starter's attempt to generate a survival predictor using
    Titanic's sinking data.
    
    This is a revamped version of my previous model
    Based on https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html and
    https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
    
    __author__ = giancarloyona
    __email__ = gianyona@gmail.com
    __date__ = jan/19
    __version__ = 0.1.2.2
"""

# importing the libraries needed for the challenge
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

# importing the data sets
training_set = pd.read_csv("../datasets/train.csv")
test_set = pd.read_csv("../datasets/test.csv")

# preprocessing the data

# extracting the target data
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

# retrieving the size of the family
full_set['FamilySize'] = full_set['SibSp'] + full_set['Parch'] + 1

# is the passenger travelling alone?
full_set['IsAlone'] = full_set['FamilySize'].map(lambda size: 0 if size > 1 else 1)
full_set.drop(labels=['SibSp', 'Parch'], inplace=True, axis=1)

# parsing the ticket data
full_set['Ticket'] = full_set['Ticket'].map(lambda ticket: ticket.replace('/', '').replace('.', '').split()[0])
full_set['Ticket'] = full_set['Ticket'].map(lambda ticket: 'U' if ticket.isdigit() else ticket[0])

# imputing missing data
imputer_age = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_age = imputer_age.fit(np.reshape(full_set.iloc[:, 2].values, (-1, 1)))
full_set['Age'] = imputer_age.transform(np.reshape(full_set.iloc[:, 2].values, (-1, 1)))

imputer_fare = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_fare = imputer_fare.fit(np.reshape(full_set.iloc[:, 4].values, (-1, 1)))
full_set['Fare'] = imputer_fare.transform(np.reshape(full_set.iloc[:, 4].values, (-1, 1)))

imputer_cabin = SimpleImputer(missing_values=np.nan, fill_value="U", strategy='constant')
imputer_cabin = imputer_cabin.fit(np.reshape(full_set.iloc[:, 5].values, (-1, 1)))
full_set['Cabin'] = imputer_cabin.transform(np.reshape(full_set.iloc[:, 5].values, (-1, 1)))
full_set['Cabin'] = full_set['Cabin'].map(lambda c: c[0])

imputer_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer_embarked = imputer_embarked.fit(np.reshape(full_set.iloc[:, 6].values, (-1, 1)))
full_set['Embarked'] = imputer_embarked.transform(np.reshape(full_set.iloc[:, 6].values, (-1, 1)))

# encoding the categorical variables
label_encoder = LabelEncoder()
full_set['Pclass'] = label_encoder.fit_transform(full_set['Pclass'])
full_set['Sex'] = label_encoder.fit_transform(full_set['Sex'])
full_set['Ticket'] = label_encoder.fit_transform(full_set['Ticket'])
full_set['Cabin'] = label_encoder.fit_transform(full_set['Cabin'])
full_set['Embarked'] = label_encoder.fit_transform(full_set['Embarked'])
full_set['Title'] = label_encoder.fit_transform(full_set['Title'])

column_tranformer = ColumnTransformer(
    [('Pclass', OneHotEncoder(categories='auto'), [0]),
     ('Sex', OneHotEncoder(categories='auto'), [1]),
     ('Ticket', OneHotEncoder(categories='auto'), [3]),
     ('Cabin', OneHotEncoder(categories='auto'), [5]),
     ('Embarked', OneHotEncoder(categories='auto'), [6]),
     ('Title', OneHotEncoder(categories='auto'), [7])
     ], remainder='passthrough', sparse_threshold=0)

full_set = column_tranformer.fit_transform(full_set)
training_set = pd.DataFrame(full_set[:891])
test_set = pd.DataFrame(full_set[891:])


# splitting the dataset
def generate_oof(classifier, x_train, y_train, x_test, k_Fold):
    """
        Helper function to generate out-of-fold predictions
    
        __args__ : [classifier, training_set, target, test_set, kFold]
    """

    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((k_Fold.get_n_splits(x_train), x_test.shape[0]))

    for i, (train, test) in enumerate(k_Fold.split(x_train)):
        x_training = x_train.iloc[train]
        y_training = y_train.iloc[train]
        x_te = x_train.iloc[test]

        classifier.fit(x_training, y_training)

        oof_train[test] = classifier.predict(x_te)
        oof_test_skf[i, :] = classifier.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# creating/fitting the models
extra_tree = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(training_set, target)
gradient_boost = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(training_set, target)
random_forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(training_set, target)

# selecting the most adequate features for the model
et_model = SelectFromModel(extra_tree, prefit=True)
et_train = et_model.transform(training_set)
et_test = et_model.transform(test_set)

gbc_model = SelectFromModel(gradient_boost, prefit=True)
gbc_train = gbc_model.transform(training_set)
gbc_test = gbc_model.transform(test_set)

rf_model = SelectFromModel(random_forest, prefit=True)
rf_train = rf_model.transform(training_set)
rf_test = rf_model.transform(test_set)

# creating/fitting the models with the optimal variables
extra_tree = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(et_train, target)
gradient_boost = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(gbc_train, np.ravel(target))
random_forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(rf_train, np.ravel(target))

# generating the results
kf = KFold(10, True)
et_train, et_test = generate_oof(extra_tree,
                                 pd.DataFrame(et_train),
                                 target,
                                 pd.DataFrame(et_test),
                                 kf)

gbc_train, gbc_test = generate_oof(gradient_boost,
                                   pd.DataFrame(gbc_train),
                                   target,
                                   pd.DataFrame(gbc_test),
                                   kf)

rf_train, rf_test = generate_oof(random_forest,
                                 pd.DataFrame(rf_train),
                                 target,
                                 pd.DataFrame(rf_test),
                                 kf)

# ensembling the models
x_train = np.concatenate((et_train, gbc_train, rf_train), axis=1)
x_test = np.concatenate((et_test, gbc_test, rf_test), axis=1)

base_predictions_train = pd.DataFrame({
    'ExtraTree': et_train.ravel(),
    'GradientBoost': gbc_train.ravel(),
    'RandomForest': rf_train.ravel()
})

gbm = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    min_child_weight=2,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objectve='binary:logistic',
    nthread=-1,
    scale_pos_weight=1).fit(x_train, target)

y_pred = gbm.predict(x_test)

# exporting the results
index = np.reshape(np.arange(start=892, stop=1310), (-1, 1))
results = np.append(index, np.reshape(y_pred, (-1, 1)), axis=1)
results = pd.DataFrame(results, columns=['PassengerId', 'Survived'])
results.to_csv('../datasets/results_xgb.csv', sep=',', index=False)
