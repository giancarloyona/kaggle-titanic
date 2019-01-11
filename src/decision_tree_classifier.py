"""
    Kaggle Getting Started Prediction Competition

    This is a starter's attempt to generate a survival predictor using 
    Titanic's sinking data.
    
    This is a revamped version of the previous model 
    Based on the work of Mr. Ahmed (https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html)
    
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
"""

title = set()

for name in full_dataset['Name']:
    title.add(name.split(',')[1].split('.')[0].strip())

""" 
full_dataset['Title'] = full_dataset['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
full_dataset.drop(labels = ['Name'], inplace = True, axis = 1)