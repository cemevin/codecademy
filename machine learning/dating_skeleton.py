import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.svm import SVC
import time

df = pd.read_csv("profiles.csv")

df["drinks_code"] = df.drinks.map({"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5})
df["smokes_code"] = df.smokes.map({"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4})
df["drugs_code"] = df.drugs.map({'never': 0, 'sometimes': 1, 'often': 2})

# most drug users are single
plt.scatter(df.status, df.drugs_code, alpha=0.01)

# mean income according to jobs
tmp = df.dropna().copy()
tmp = tmp[tmp.income > -1]
tmp = tmp[~tmp.job.isin(['rather not say'])]
meanIncome = tmp.groupby('job').mean()['income']
meanIncome.plot()

### body type detection from diet, drinking, smoking, drugs
dataset = df.dropna().copy()

# process the body type
dataset = dataset[dataset.body_type != 'rather not say']
dataset['body_code'] = dataset['body_type'].apply(lambda x: 0 if x in ['thin', 'skinny', 'used up'] else 1
           if x in ['average', 'full figured', 'fit', 'athletic', 'jacked'] else 2 )

# process diet
dataset['is_veg'] = dataset.diet.map(lambda x: 1 if x.find('veg') > -1 else 0)

dataset = dataset[['is_veg', 'body_code', 'drinks_code', 'smokes_code', 'drugs_code']]
X = dataset[['is_veg', 'drinks_code', 'smokes_code', 'drugs_code']]
y = dataset['body_code']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# SVC
time_before = time.perf_counter()
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)
print('time:' + str(time.perf_counter() - time_before))
print(clf.score(X_test, y_test))

# Decision Tree
time_before = time.perf_counter()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print('time:' + str(time.perf_counter() - time_before))
print(clf.score(X_test, y_test))

### estimate age from income, job type, drink, smoke, drugs

# one hot encoding of jobs since they aren't linearly increasing
dataset = pd.concat([df[['age','drinks_code', 'smokes_code', 'drugs_code', 'income']], pd.get_dummies(df['job'], prefix='job')], axis=1).dropna()
dataset.income = dataset.income.apply(lambda x: 0 if x == -1 else x)

X = dataset.drop(columns=['age'])
y = dataset['age']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# SVR
from sklearn.svm import SVR
time_before = time.perf_counter()
regr = SVR()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
print('time:' + str(time.perf_counter() - time_before))

# KNN
from sklearn.neighbors import KNeighborsRegressor
time_before = time.perf_counter()
for i in range(1,21):
    neigh = KNeighborsRegressor(n_neighbors=i)
    neigh.fit(X_train, y_train)
    print(neigh.score(X_test, y_test))
print('time:' + str(time.perf_counter() - time_before))