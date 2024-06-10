from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy
import numpy as np
import pandas as pd
from matplotlib.cm import rainbow
from matplotlib.colors import ListedColormap
import seaborn as sns
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from collections import Counter
import time
import queue
start_time = time.time()
#%matplotlib inline
filename = 'TrainingHistorydemo.csv'
Features = ['sno','freq','output']
data = pd.read_csv(filename, names=Features)
data.head(3)
print("--- %s seconds ---" % (time.time() - start_time))
pd.value_counts(data['output']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('output')
plt.ylabel('Frequency')
data['output'].value_counts()
from sklearn.preprocessing import StandardScaler
data['norm'] = StandardScaler().fit_transform(data['freq'].values.reshape(-1, 1))
data = data.drop(['freq'], axis=1)
data.head()
X = np.array(data.loc[:, data.columns != 'output'])
y = np.array(data.loc[:, data.columns == 'output'])

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

### Fit a sklearn classifier on train dataset and output probabilities
clf = RandomForestClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]
### Compute ROC curve and ROC area for predictions on validation set
fpr1, tpr1, _ = roc_curve(y_test, pred_val)
roc_auc1 = auc(fpr1, tpr1)
predictions = model.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 
# print classification report 
#print(classification_report(y_test, pred_val)) 

clf = ExtraTreesClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr2, tpr2, _ = roc_curve(y_test, pred_val)
roc_auc2 = auc(fpr2, tpr2)

clf = CatBoostClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr3, tpr3, _ = roc_curve(y_test, pred_val)
roc_auc3 = auc(fpr3, tpr3)

clf = XGBClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr4, tpr4, _ = roc_curve(y_test, pred_val)
roc_auc4 = auc(fpr4, tpr4)

clf = AdaBoostClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr5, tpr5, _ = roc_curve(y_test, pred_val)
roc_auc5 = auc(fpr5, tpr5)

clf = lgb.LGBMClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr6, tpr6, _ = roc_curve(y_test, pred_val)
roc_auc6 = auc(fpr6, tpr6)

### Plot
plt.figure()
lw = 2
plt.plot(fpr1, tpr1,lw=lw, label='RF ROC curve (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2,lw=lw, label='ET ROC curve (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3,lw=lw, label='CAT ROC curve (area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4,lw=lw, label='XGB ROC curve (area = %0.2f)' % roc_auc4)
plt.plot(fpr5, tpr5,lw=lw, label='ADB ROC curve (area = %0.2f)' % roc_auc5)
plt.plot(fpr6, tpr6,lw=lw, label='LGB ROC curve (area = %0.2f)' % roc_auc6)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for single user using SMOTE oversampling')
plt.legend(loc="lower right")
plt.show()


from imblearn.under_sampling import NearMiss
trans = NearMiss(version=1)
from imblearn.over_sampling import ADASYN

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)  
print("Number transactions y_test dataset: ", y_test.shape)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

X_train_res, y_train_res = trans.fit_resample(X_train, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))



from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

### Fit a sklearn classifier on train dataset and output probabilities
clf = RandomForestClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr1, tpr1, _ = roc_curve(y_test, pred_val)
roc_auc1 = auc(fpr1, tpr1)
predictions = model.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions)) 


clf = ExtraTreesClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr2, tpr2, _ = roc_curve(y_test, pred_val)
roc_auc2 = auc(fpr2, tpr2)

clf = CatBoostClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr3, tpr3, _ = roc_curve(y_test, pred_val)
roc_auc3 = auc(fpr3, tpr3)

clf = XGBClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr4, tpr4, _ = roc_curve(y_test, pred_val)
roc_auc4 = auc(fpr4, tpr4)

clf = AdaBoostClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr5, tpr5, _ = roc_curve(y_test, pred_val)
roc_auc5 = auc(fpr5, tpr5)

clf = lgb.LGBMClassifier()
model = clf.fit(X_train_res, y_train_res)
pred_val = model.predict_proba(X_test)[:,1]

### Compute ROC curve and ROC area for predictions on validation set
fpr6, tpr6, _ = roc_curve(y_test, pred_val)
roc_auc6 = auc(fpr6, tpr6)

### Plot
plt.figure()
lw = 2
plt.plot(fpr1, tpr1,lw=lw, label='RF ROC curve (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2,lw=lw, label='ET ROC curve (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3,lw=lw, label='CAT ROC curve (area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4,lw=lw, label='XGB ROC curve (area = %0.2f)' % roc_auc4)
plt.plot(fpr5, tpr5,lw=lw, label='ADB ROC curve (area = %0.2f)' % roc_auc5)
plt.plot(fpr6, tpr6,lw=lw, label='LGB ROC curve (area = %0.2f)' % roc_auc6)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for single user using near miss undersampling')
plt.legend(loc="lower right")
plt.show()