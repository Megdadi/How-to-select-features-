# -*- coding: utf-8 -*-
"""
Feature selection : Reduce the number of input features by the effect
of each feature on the target variable.
1. sklearn.feature_selection.SelectPercentile
2. GenericUnivariateSelect
3. Select From Model
"""
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Matlab-style plotting
import seaborn as sns # making statistical graphics in Python
import numpy as np # linear algebra
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
#### load breast cancer data
BreastData = load_breast_cancer()
features=BreastData.data
aaaa=features[:,0]# first column
aaaa=features[:,1]# second column
feature_names= BreastData.feature_names
print('features shape is ' , features.shape)
numberofcolumns= len(features[0]) #  number of columns
target=BreastData.target
target_names= BreastData.target_names # malignant:0, benign:1
# ###############  Correlation matrix (heatmap style) ###########
# from scipy.stats import pearsonr
# correlation=[]
# i=0
# for i in range(numberofcolumns):
#     corr, _ = pearsonr(features[:,i], target)
#     correlation.append(corr)
 
# correlation=pd.DataFrame(correlation)
# # correlation.sort_values(0,ascending=True)
# f, ax = plt.subplots(figsize=(20, 20))
# sns.set(font_scale=1.25)
# hm = sns.heatmap(correlation, cbar=True, annot=True, square=True)
# plt.show()

############### 1. selection.SelectPercentile ##############################3
# from sklearn.feature_selection import SelectPercentile, chi2 , f_classif 

# FeatureSelection = SelectPercentile(score_func = chi2, percentile=20) # score_func can = f_classif
# features = FeatureSelection.fit_transform(features, target) # number of features become
# print('features shape is ' , features.shape)
# features=pd.DataFrame(features)
# print('Selected Features are : ' , FeatureSelection.get_support())

############# 2. GenericUnivariateSelect #########################
# from sklearn.feature_selection import GenericUnivariateSelect
# from sklearn.feature_selection import chi2 , f_classif 
# FeatureSelection = GenericUnivariateSelect(score_func= chi2, mode= 'k_best', param=6) # score_func can = f_classif : mode can = percentile,fpr,fdr,fwe 
# features = FeatureSelection.fit_transform(features, target)
# print('features shape is ' , features.shape)

# print('Selected Features are : ' , FeatureSelection.get_support())

############# 3. SelectFromModel #######################################
####### model:LinearRegression
# features.shape
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LinearRegression

# thismodel = LinearRegression()
# FeatureSelection = SelectFromModel(estimator = thismodel, max_features = None) # make sure that thismodel is well-defined
# features = FeatureSelection.fit_transform(features, target)
# print('features shape is ' , features.shape)

# print('Selected Features are : ' , FeatureSelection.get_support())

####### model:RandomForest
features.shape
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

FeatureSelection = SelectFromModel(RandomForestClassifier(n_estimators = 20))  # make sure that thismodel is well-defined
features = FeatureSelection.fit_transform(features, target)
print('features shape is ' , features.shape)

print('Selected Features are : ' , FeatureSelection.get_support())