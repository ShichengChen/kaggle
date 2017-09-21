import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC


def featureEngineering():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_len = len(train)
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
    dataset = dataset.fillna(np.nan)

    dataset['SibSp0'] = dataset['SibSp'].map(lambda s: 1 if s == 0 else 0)
    dataset['SibSp1'] = dataset['SibSp'].map(lambda s: 1 if s == 1  else 0)
    dataset['SibSp2'] = dataset['SibSp'].map(lambda s: 1 if s >= 2 else 0)
    dataset['Parch0'] = dataset['SibSp'].map(lambda s: 1 if s == 0 else 0)
    dataset['Parch1'] = dataset['SibSp'].map(lambda s: 1 if s == 1  else 0)
    dataset['Parch2'] = dataset['SibSp'].map(lambda s: 1 if s == 2 else 0)
    dataset['Parch3'] = dataset['SibSp'].map(lambda s: 1 if s >= 3 else 0)

    bins = [-10, 0, 10, 100]
    dataset['Age_map'] = dataset['Age'].copy()
    dataset['Age_map'] = dataset['Age_map'].fillna(-3)
    dataset['Age_map'] = pd.cut(dataset[:train_len]['Age_map'], bins=bins)

    dataset = pd.get_dummies(dataset, columns=["Age_map"], prefix="Age")
    dataset.drop(['Age_(10, 100]'], axis=1, inplace=True)

    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"]
    dataset['Fsize0'] = dataset['Fsize'].map(lambda s: 1 if s == 0 else 0)
    dataset['Fsize1'] = dataset['Fsize'].map(lambda s: 1 if s == 1  else 0)
    dataset['Fsize2'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
    dataset['Fsize3'] = dataset['Fsize'].map(lambda s: 1 if s == 3  else 0)
    dataset['Fsize4'] = dataset['Fsize'].map(lambda s: 1 if s >= 4  else 0)

    median = dataset[(dataset.Pclass == 3) & (dataset.Embarked == 'S')]['Fare'].median()
    dataset["Fare"] = dataset["Fare"].fillna(median)
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    dataset.Fare = dataset.Fare.astype(int)

    dataset['namel'] = dataset['Name'].apply(lambda x: len(x))
    bins = [10, 20, 30, 40, 50, 100]
    dataset['name_map'] = dataset['namel'].copy()
    dataset['name_map'] = pd.cut(dataset[:train_len]['name_map'], bins=bins)
    dataset = pd.get_dummies(dataset, columns=["name_map"], prefix="name")
    dataset.drop(['name_(20, 30]'], axis=1, inplace=True)

    dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(dataset_title)
    mapt = {"Miss": 'girl', "Mlle": 'girl',
            'Don': 'men', "Mr": 'men', 'Sir': 'men', 'Major': 'men', 'Rev': 'men',
            'Dr': 'men', 'Col': 'men', 'Jonkheer': 'men', 'Capt': 'men',
            'the Countess': 'women', 'Dona': 'women', 'Lady': 'women', "Ms": 'women', "Mme": 'women', "Mrs": 'women',
            'Master': 'Master'
            }
    dataset["Title"] = dataset["Title"].map(mapt)
    dataset = pd.get_dummies(dataset, columns=["Title"])

    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
    dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
    dataset.drop(labels=["Cabin_G", "Cabin_T", 'Cabin_F', 'Cabin_A'], axis=1, inplace=True)

    Ticket = []
    for i in list(dataset.Ticket):
        if not i.isdigit():
            Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
        else:
            Ticket.append(str(i)[0])
    dataset["Ticket"] = Ticket
    dataset["Ticket"] = pd.Series([i[0] for i in dataset['Ticket']])
    dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
    dataset.drop(labels=["T_W", "T_F", 'T_L', 'T_S', 'T_C'], axis=1, inplace=True)
    dataset.drop(labels=["T_4", "T_5", 'T_6', 'T_7', 'T_8', 'T_9'], axis=1, inplace=True)

    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})

    dataset["Embarked"] = dataset["Embarked"].fillna('C')
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    dataset.drop(labels=['Em_Q'], axis=1, inplace=True)

    dataset["Pclass"] = dataset["Pclass"].astype('category')
    dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")




    dataset.drop(labels=["PassengerId", "Name"], axis=1, inplace=True)
    dataset.drop(labels=['SibSp', 'Fsize', 'Parch','Age','namel'], axis=1, inplace=True)

    col = list(dataset.columns.values)
    ind1 = col.index('Survived')
    col[ind1], col[0] = col[0], col[ind1]
    dataset = dataset[col]
    dataset.to_csv('featureEngineering.csv')
    return dataset

print featureEngineering().columns.values
