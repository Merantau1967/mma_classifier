# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:17:46 2018

@author: Tigor
"""
import pandas

import pandas
import scipy
import numpy
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pyodbc as db
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler


con = db.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=HP-LAPTOP\SQLSERVER2014;Trusted_Connection=yes;DATABASE=mma_magic')
            

dataset = pandas.read_sql(('''SELECT 
    --   average_way_of_finish
    --  ,average_round_finish
    --  ,average_time_finish
    --  ,
    -- fighter_weight
    --   ,fighter_age
    --  ,fighter_height
    --  ,average_way_of_finish_o
    --  ,average_round_finish_o
    --  ,average_time_finish_o
     --   ,fighter_weight_o
    --  ,fighter_age_o
    --  ,fighter_height_o
    -- , 
     LOSSES_DECISIONS
    ,LOSSES_KOTKO
    ,LOSSES_SUBMISSIONS
   , WINS_DECISIONS
    ,WINS_KOTKO
    ,WINS_SUBMISSIONS
    ,LOSSES_DECISIONS_o
    ,LOSSES_KOTKO_o
    ,LOSSES_SUBMISSIONS_o
    ,WINS_DECISIONS_o
    ,WINS_KOTKO_o
    ,WINS_SUBMISSIONS_o
    ,win_loss
  FROM dbo.mma_stats_main_v'''),con)

    
# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('win_loss').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False)
plt.show()

 #histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

dataset.fillna(0)
#dataset['losses_submissions']=dataset['losses_submissions'].fillna(0)
#dataset['wins_desisions']=dataset['wins_desisions'].fillna(0)
#dataset['losses_submissions']=dataset['losses_submissions'].fillna(0)
#dataset['wins_desisions']=dataset['wins_desisions'].fillna(0)


print(dataset.describe())
dataset.head()
#print (dataset.isnull())


# split training en test set
array = dataset.values
X = array[:,0:12]
Y = array[:,12]

print(X)


X1 = array[:,2]
X2 = array[:,11]
scaler = MinMaxScaler(feature_range=(0, 1))
#rescaledX1 = scaler.fit_transform(X1)
#rescaledX2 = scaler.fit_transform(X2)
#rescaledY1 = scaler.fit_transform(Y)


print (X1)
print (Y)

#print(X)

rng = numpy.random.RandomState(0)

colors = rng.rand(100)
sizes = 1000 * rng.rand(100)


#scatter(X1,X2,Y,"o")
plt.scatter(X1,X2, c=Y, s=5, alpha=1,
            cmap='viridis')
plt.colorbar();  # show color scale




print(Y)
Y=Y.astype('int')
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

print("rescaled....",rescaledX[0:5,:])

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(rescaledX, Y, test_size=validation_size, random_state=seed)

#10 - fold validation

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('DM', DummyClassifier(strategy="uniform")))
models.append(('DM1', DummyClassifier(strategy="most_frequent")))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3,  weights = 'distance')))
#models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('SVM', SVC()))
models.append(('NN', MLPClassifier(alpha=1)))
models.append(('RF',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #cv_results = model_selection.cross_val_score(model, rescaledX, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg) 


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset

#validation_size = 0.20
#seed = 7
#X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(rescaledX, Y, test_size=validation_size, random_state=seed)

#print(X_validation)

results = []
acc_score = []
names=[]

for k in range(1,20):
    
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
    #knn.fit(X_train, Y_train)
    #predictions = knn.predict(X_validation)
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(knn, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(k)
    #print(accuracy_score(Y_validation, predictions))
    #acc_score.append(accuracy_score(Y_validation, predictions))
    #results.append(cv_results)
    #print(cv_results)

print( cv_results )

fig = plt.figure()
fig.suptitle('different K')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


knn = KNeighborsClassifier(n_neighbors=3,  weights = 'distance')
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)


#print("knn------------------->", predictions)
    
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions)) 




dataset_real = pandas.read_sql(('''SELECT 
    --   average_way_of_finish
    --  ,average_round_finish
    --  ,average_time_finish
    --  ,
    -- fighter_weight
    --   ,fighter_age
    --  ,fighter_height
    --  ,average_way_of_finish_o
    --  ,average_round_finish_o
    --  ,average_time_finish_o
     --   ,fighter_weight_o
    --  ,fighter_age_o
    --  ,fighter_height_o
    -- , 
     LOSSES_DECISIONS
    ,LOSSES_KOTKO
    ,LOSSES_SUBMISSIONS
   , WINS_DECISIONS
    ,WINS_KOTKO
    ,WINS_SUBMISSIONS
    ,LOSSES_DECISIONS_o
    ,LOSSES_KOTKO_o
    ,LOSSES_SUBMISSIONS_o
    ,WINS_DECISIONS_o
    ,WINS_KOTKO_o
    ,WINS_SUBMISSIONS_o
    ,win_loss
    ,fighter_name
    ,fighter_name_o
  FROM dbo.mma_stats_main_v_real_dynamic'''),con)

array = dataset_real.values
X_real = array[:,0:12]

X_names = array[:,13:15]

print(X_names)

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_real = scaler.fit_transform(X_real)

print("=====================Let's go!!!==========================")

#-- NOW PREDICT! -----------------------------------------

#print("X real", X_real)
#print("rescaledX real", rescaledX_real)
#print("X",X)
#print("Y",Y)

knn = KNeighborsClassifier(n_neighbors=10,   weights = 'distance')
#knn.fit(rescaledX, Y)
knn.fit(X, Y)
predictions = knn.predict(X_real)

print("knn", predictions)


# Make predictions on validation dataset

nb = GaussianNB()
nb.fit(X, Y)
predictions = nb.predict(X_real)

print("GaussianNB", predictions)


dt =  DecisionTreeClassifier()
dt.fit(X, Y)
predictions = dt.predict(X_real)

print("DecisionTreeClassifier", predictions)


ld = LinearDiscriminantAnalysis()

ld.fit(X, Y)
predictions = ld.predict(X_real)

print("LinearDiscriminantAnalysis",predictions)

an = MLPClassifier(alpha=0.5)
an.fit(X, Y)

predictions = an.predict(X_real)

print("MLPClassifier",predictions)

rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rf.fit(X, Y)

predictions = rf.predict(X_real)

print("RandomForestClassifier", predictions)





print("===================== einde ==========================")




