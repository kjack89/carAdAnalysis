# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:16:36 2020

@author: kcjac
"""

import math
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plot

#choose save location
csvSavePath = 'C:\\Users\\Kevin\\Desktop\\consDataset\\'

#returns the accuracy of XGBoost model. If get_booster is True, returns the actual model to be used for future predictions
def XGBmodelfit(alg, dtrain, predictors,target,useTrainCV=True, cv_folds=5, early_stopping_rounds=50, get_booster = False):
   
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
   
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='mlogloss')
    
    if get_booster is False:
        #Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        
        #Print model report:
        print(target)
        print("XGBoost Model Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
        feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
    else:
        return alg.get_booster()
    
                      
def KNNmodelfit(alg, dtrain, predictors,target,useTrainCV=True, cv_folds=5):
   
    if useTrainCV:
        yloc = dtrain.columns.get_loc(target)
        scale_fitKNN = StandardScaler().fit_transform(dtrain) #normalize features for Knn
        y = dtrain[target]
        X = np.delete(scale_fitKNN, obj=yloc, axis=1)
   
    #Fit the algorithm on the data
    alg.fit(X, y)
       
    #Predict training set:
    dtrain_predictions = alg.predict(X)
       
    #Print model report:
    print(target)
    print("Knn Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))

def SVMmodelfit(alg, dtrain, predictors,target,useTrainCV=True, cv_folds=5):
   
    if useTrainCV:
        yloc = dtrain.columns.get_loc(target)
        scale_fitSVM = StandardScaler().fit_transform(dtrain) #normalize features for Knn
        y = dtrain[target]
        X = np.delete(scale_fitSVM, obj=yloc, axis=1)
   
    #Fit the algorithm on the data
    alg.fit(X, y)
       
    #Predict training set:
    dtrain_predictions = alg.predict(X)
       
    #Print model report:
    print(target)
    print("SVM Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))
    
#returns dictionary of mapped dataframe values that map classes to numeric type
def mapValues(df):
    Dict = {}
    for col in df.columns:
        tempset = set()
        for val in df[col].values:
            if val is None:
                continue
            elif type(val) != str and math.isnan(val):
                continue
            else:
                tempset.add(val)
        mapVals = {}
        for ind, val in enumerate(tempset):
            mapVals[val] = ind
        Dict[col] = mapVals
    return(Dict)

#transforms data similar to sclearn.preprocessing LabelEncoder, but skips null values
def transformData(df, dic):
    newdf = df
    for col in newdf.columns:
        for index,val in zip(newdf.index,newdf[col].values):
            if val is None:
                continue
            elif type(val) != str and math.isnan(val):
                continue
            else:
                mappings = dic[col]
                for key in mappings.keys():
                    if key == val:
                        newdf.at[index,col] = mappings[key]
        newdf = newdf.astype({col:'float64'})
    return newdf

#reverses transformation of transformData back to the class names
def inverse_transformData(df,dic):
    newdf = df
    for col in newdf.columns:
        if type(next(iter(dic[col]))) == str:
            newdf = newdf.astype({col:'object'})
        for index,val in zip(newdf.index,newdf[col].values):
            if val is None:
                continue
            elif type(val) != str and math.isnan(val):
                continue
            else:
                mappings = dic[col]
                for key in mappings.keys():
                    if mappings[key] == val:
                        newdf.at[index,col] = key
    return newdf


cars = pd.read_csv(csvSavePath+'cleanedCars.csv')    
train = cars[(cars['type'].notnull())&(cars['condition'].notnull())&(cars['title_status'].notnull())&cars['size'].notnull()&
        (cars['drive'].notnull())&(cars['transmission'].notnull())&(cars['cylinders'].notnull())&(cars['fuel'].notnull())&
        (cars['model'].notnull())&(cars['make'].notnull())&(cars['year'].notnull())]
cols = ['type', 'size', 'drive' , 'transmission', 'fuel', 'model', 'make', 'cylinders','title_status','year','condition']
train = train[cols]
makeTrain = train[['type', 'size', 'drive' , 'transmission', 'fuel', 'make', 'cylinders','title_status','year','condition']]
makecols = ['type', 'size', 'drive' , 'transmission', 'fuel', 'make', 'cylinders','title_status','year','condition']

dic = mapValues(cars) 
train1 = transformData(train, dic)

import warnings
warnings.filterwarnings("ignore")
#run to compare between XGBoost, Knn, and SVM for each feature to impute
#these are vanilla model versions, different results can be attained with parameter tuning
for col in cols:
    target = col
    predictors = list(set(cols) - set([col]))
    knn1 = KNeighborsClassifier(n_neighbors = 3)
    svm1 = LinearSVC()
    xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        num_class = len(train[target].unique()),
        objective= 'multi:softmax',
        nthread=4,
        scale_pos_weight=1,
        seed=7)
    XGBmodelfit(xgb1, train1, predictors, target)
    print('\n')
    KNNmodelfit(knn1, train1, predictors, target)
    print('\n')
    SVMmodelfit(svm1, train1, predictors, target)
    print('\n')

types = cars[cars['type'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

cylinders = cars[cars['cylinders'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['type'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

size = cars[cars['size'].isnull()&cars['type'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

drive = cars[cars['drive'].isnull()&cars['size'].notnull()&cars['type'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

transmission = cars[cars['transmission'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['type'].notnull()&cars['type'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

fuel = cars[cars['fuel'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()]

title_status = cars[cars['title_status'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['model'].notnull()&cars['make'].notnull()&cars['cylinders'].notnull()&cars['type'].notnull()&cars['year'].notnull()]

make = cars[cars['make'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['type'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()&cars['condition'].notnull()]

condition = cars[cars['condition'].isnull()&cars['size'].notnull()&cars['drive'].notnull()&cars['transmission'].notnull()&cars['fuel'].notnull()&
        cars['type'].notnull()&cars['cylinders'].notnull()&cars['title_status'].notnull()&cars['year'].notnull()&cars['make'].notnull()]

print(cars.isnull().sum())
print('types: '+str(types.shape))
print('cylinders: '+str(cylinders.shape))
print('size: '+str(size.shape))
print('drive: '+str(drive.shape))
print('transmission: '+str(transmission.shape))
print('fuel: '+str(fuel.shape))
print('title_status: '+str(title_status.shape))
print('make: '+str(make.shape))
print('condition'+str(condition.shape))


#check accuracy of model first for type
target = 'type'
predictors = list(set(cols) - set([target]))
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=670,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = len(train1[target].unique()),
    objective= 'multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=7)
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=True, get_booster=False)
#creates model for predictions, then imputes the null target rows for type

#transform classes to numeric types
transformedtypes = transformData(types,dic)
#create the model for predictions using the same dataset used to check for accuracy
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=False, get_booster=True)
#predict using new dataframe with null targets and put the predicted values into the new dataframe
transformedtypes[target]=np.argmax(x.predict(xgb.DMatrix(transformedtypes[predictors].values, feature_names=predictors)), axis = 1)
#change numerics back to class names in new dataframe
revertedtypes = inverse_transformData(transformedtypes,dic)
#put the imputed values back into the main dataframe
cars.loc[cars.index.isin(revertedtypes.index), target] = revertedtypes[target].values

#check accuracy of model first for drive
target = 'drive'
predictors = list(set(cols) - set([target]))
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=670,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = len(train1[target].unique()),
    objective= 'multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=7)
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=True, get_booster=False)
#creates model for predictions, then imputes the null target rows for drive

#transform classes to numeric types
transformeddrive = transformData(drive,dic)
#create the model for predictions using the same dataset used to check for accuracy
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=False, get_booster=True)
#predict using new dataframe with null targets and put the predicted values into the new dataframe
transformeddrive[target]=np.argmax(x.predict(xgb.DMatrix(transformeddrive[predictors].values, feature_names=predictors)), axis = 1)
#change numerics back to class names in new dataframe
reverteddrive = inverse_transformData(transformeddrive,dic)
#put the imputed values back into the main dataframe
cars.loc[cars.index.isin(reverteddrive.index), target] = reverteddrive[target].values

#check accuracy of model first for cylinders
target = 'cylinders'
predictors = list(set(cols) - set([target]))
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=670,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = len(train1[target].unique()),
    objective= 'multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=7)
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=True, get_booster=False)
#creates model for predictions, then imputes the null target rows for cylinders

#transform classes to numeric types
transformedcylinders = transformData(cylinders,dic)
#create the model for predictions using the same dataset used to check for accuracy
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=False, get_booster=True)
#predict using new dataframe with null targets and put the predicted values into the new dataframe
transformedcylinders[target]=np.argmax(x.predict(xgb.DMatrix(transformedcylinders[predictors].values, feature_names=predictors)), axis = 1)
#change numerics back to class names in new dataframe
revertedcylinders = inverse_transformData(transformedcylinders,dic)
#put the imputed values back into the main dataframe
cars.loc[cars.index.isin(revertedcylinders.index), target] = revertedcylinders[target].values

#check accuracy of model first for size
target = 'size'
predictors = list(set(makecols) - set([target]))
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=670,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = len(dic[target]),
    objective= 'multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=7)
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=True, get_booster=False)
#creates model for predictions, then imputes the null target rows for size

#transform classes to numeric types
transformedsize = transformData(size,dic)
#create the model for predictions using the same dataset used to check for accuracy
x = XGBmodelfit(xgb1, train1, predictors, target, useTrainCV=False, get_booster=True)
#predict using new dataframe with null targets and put the predicted values into the new dataframe
transformedsize[target]=np.argmax(x.predict(xgb.DMatrix(transformedsize[predictors].values, feature_names=predictors)), axis = 1)
#change numerics back to class names in new dataframe
revertedsize = inverse_transformData(transformedsize,dic)
#put the imputed values back into the main dataframe
cars.loc[cars.index.isin(revertedsize.index), target] = revertedsize[target].values

train2 = cars[cols].dropna()
train2 = transformData(train2,dic)
#check accuracy of model first for condition
target = 'condition'
predictors = list(set(makecols) - set([target]))
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=670,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    num_class = len(dic[target]),
    objective= 'multi:softmax',
    nthread=4,
    scale_pos_weight=1,
    seed=7)
x = XGBmodelfit(xgb1, train2, predictors, target, useTrainCV=True, get_booster=False)

def imputeOdometer(dataframecol,val):
    if np.isnan(dataframecol) or dataframecol > 500000:
        return val
    else:
        return dataframecol
    
def splithour(text):
    return(text.split(':')[0])

def splitmin(text):
    return(text.split(':')[1])

val = cars[cars['odometer'] < 500000]['odometer'].mean()    
cars['odometer'] = cars['odometer'].apply(imputeOdometer,val=val)
cars['date'] = cars['date'].astype('datetime64')
cars['posting_hour'] = cars['time'].apply(splithour)
cars['posting_min'] = cars['time'].apply(splitmin)
cars['year'] = cars['year'].astype('int32')
cars['odometer'] = cars['odometer'].astype('int32')
cars['posting_hour'] = cars['posting_hour'].astype('int32')
cars['posting_min'] = cars['posting_min'].astype('int32')

cleanedimputedcars = cars[['price','year','make','model','odometer','title_status','condition','type','size','drive','cylinders','fuel','transmission','color','city','state','date','posting_hour','posting_min']]
cleanedimputedcars.to_csv(csvSavePath+'imputedcleanedCars.csv')


