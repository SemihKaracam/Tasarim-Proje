

## 7 farklı veri seti dengeleme yöntemi kullanılarak indirgenmis veri seti dengelendi.
## Dengelenmiş veri seti ile hiper parametre optimizasyonu yapılan sınıflandırma modelleri eğitildi.
## Dengeleme yöntemleri ve sınıflandırma modellerinin performansları karşılaştırıldı.
## Ensemble Öğrenme teknikleri de kullanıldı. (en altta)


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report, auc
from yellowbrick.classifier import ClassificationReport

## Techniques to Convert Imbalanced Dataset into Balanced Dataset With Random Forest Classifier¶
## 1) Random Oversampling with Evaluation
## 1.1) With SMOTE (Synthetic Minority Oversampling Technique) with Evaluation
## 1.2) ADASYN Random over-sampling Evaluation
## 2) Random Under-Sampling With Evaluation
## 2.1) With Near Miss Under-Sampling With Evaluation
## 3) Ensemble Learning Techniques with Evaluation
## 3.1 ) Bagging classifier
## 3.2) Boosting
## 3.3) Forest of randomized trees
## 4) Combination of over- and under-sampling With Evaluation
## 4.1) SMOTEENN
## 4.2) SMOTETomek

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE    #pip3 install imblearn
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek


# Duzenlenmis veri seti icin
data = pd.read_csv("data.csv")
df = data.copy()

# Orijinal veri seti icin
# df = pd.read_csv("data0.csv")


# df.head()
# df.info()
# df.describe()


# Sınıflar arası dengesizliğin görsellestirilmesi

# df2 = df['Heart_attack_risk'].value_counts().reset_index(name='count')
# plt.figure(figsize = (8,8))
# plt.pie(df2['count'], labels=['Kalp krizi riski yok','Kalp krizi riski var'], colors=["#ADA190","#BB0880"], autopct='%.0f%%')
# plt.show()



y = df.Heart_attack_risk
X = df.drop(["Heart_attack_risk"], axis =1).astype("float64")

#normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)





# #################### LOGİSTİC REGRESSION #############################################################################################################


# ## RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_LR.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_ros, y_pred_ros) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_LR.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_smote, y_pred_smote) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_LR.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_LR.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_rus, y_pred_rus) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_LR.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_LR = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_LR = hyper_LR.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_smote_enn_LR, y_pred_smote_enn_LR) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_enn_LR, y_pred_smote_enn_LR), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# hyper_LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# hyper_LR.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek =hyper_LR.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized Logistic Regression:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Logistic Regression:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# #################### NAİVE BAYES #############################################################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_NB.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_ros, y_pred_ros) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# # SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_NB.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_smote, y_pred_smote) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# # ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_NB.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_NB.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_rus, y_pred_rus) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09) 
# hyper_NB.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_NB.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_NB = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_NB = hyper_NB.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_smote_enn_NB, y_pred_smote_enn_NB) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_enn_NB, y_pred_smote_enn_NB), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# hyper_NB = GaussianNB(priors= None, var_smoothing= 1e-09)
# hyper_NB.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek = hyper_NB.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized Naive Bayes:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Naive Bayes:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# #################### RANDOM FOREST #############################################################################################################


# ## RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_RF.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_RF.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_ros, y_pred_ros) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Random Forest:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_RF.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_RF.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_smote, y_pred_smote) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Random Forest:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_RF.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_RF.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Random Forest:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_RF.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_RF.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_rus, y_pred_rus) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Random Forest:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_RF.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_RF.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Random Forest:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# # plt.show()



# # ## SMOTEENN
# # smote_enn = SMOTEENN(random_state=42)
# # x_ST, y_ST = smote_enn.fit_resample(X, y)
# # x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_RF = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# # hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# # hyper_RF.fit(x_train_smote_enn,y_train_smote_enn)

# # y_pred_smote_enn_RF = hyper_RF.predict(x_test_smote_enn)
# # print('\n\nEvaluation with SMOTEENN:')
# # print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_smote_enn_RF, y_pred_smote_enn_RF) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Random Forest:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_enn_RF, y_pred_smote_enn_RF), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # ## SMOTETomek
# # smote_tomek = SMOTETomek(random_state=42)
# # x_SK, y_SK = smote_tomek.fit_resample(X, y)
# # x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# # hyper_RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# # hyper_RF.fit(x_train_smote_tomek,y_train_smote_tomek)

# # y_pred_smote_tomek = hyper_RF.predict(x_test_smote_tomek)
# # print('\n\nEvaluation with SMOTETomek:')
# # print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Random Forest:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # ####################### XGB ##############################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_ros,y_train_ros)

# y_pred_ros = xgb.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_ros, y_pred_ros) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_smote,y_train_smote)

# y_pred_smote = xgb.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_smote, y_pred_smote) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = xgb.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_rus,y_train_rus)

# y_pred_rus = xgb.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_rus, y_pred_rus) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = xgb.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_XGB = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_XGB = xgb.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_smote_enn_XGB, y_pred_smote_enn_XGB) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_enn_XGB, y_pred_smote_enn_XGB), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                      reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5) #Extreme Gradient Boost
# xgb.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek = xgb.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized XGB:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized XGB:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ################################################### KNN #############################################################

# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_knn.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_ros, y_pred_ros) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_knn.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_smote, y_pred_smote) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_knn.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_knn.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_rus, y_pred_rus) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_knn.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_KNN = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_KNN = hyper_knn.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_smote_enn_KNN, y_pred_smote_enn_KNN) )


# # Classification Report with Yellow Brick
# visualizer = ClassificationReport(hyper_knn, classes=["healthy", "unhealthy"])  # Replace with your class labels
# visualizer.fit(x_train_smote_enn, y_train_smote_enn)
# visualizer.score(x_test_smote_enn, y_test_smote_enn_KNN)
# visualizer.show()


# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_enn_KNN, y_pred_smote_enn_KNN), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# hyper_knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='uniform')
# hyper_knn.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek = hyper_knn.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized KNN:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# # Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized KNN:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# ####################################### DECISION TREE #######################################################################


# # RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_DT.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_ros, y_pred_ros) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Decision Tree:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_DT.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_smote, y_pred_smote) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Decision Tree:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_DT.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# ## Karmaşıklık matrisi
# print("Confussion matrix of Hyper optimized Decision Tree:")
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# plt.xlabel('Tahmin Edilen Sınıf')
# plt.ylabel('Gerçek Sınıf')
# plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_DT.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_rus, y_pred_rus) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Decision Tree:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf') 
# # plt.show()


# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_DT.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Decision Tree:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_DT = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_DT = hyper_DT.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_smote_enn_DT, y_pred_smote_enn_DT) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Decision Tree:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_enn_DT, y_pred_smote_enn_DT), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# hyper_DT =  DecisionTreeClassifier(criterion = 'gini',max_depth = 5,max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# hyper_DT.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek = hyper_DT.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized Decision Tree:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# # # Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized Decision Tree:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()



# #################### SVM #############################################################################################################


# ## RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# x_ros, y_ros = ros.fit_resample(X, y)
# x_train_ros, x_test_ros, y_train_ros, y_test_ros = train_test_split(x_ros, y_ros, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_ros,y_train_ros)

# y_pred_ros = hyper_SVC.predict(x_test_ros)
# print('\n\nEvaluation with RandomOverSampler:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_ros, y_pred_ros) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_ros, y_pred_ros), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()



# ## SMOTE
# sm = SMOTE(random_state = 42)
# x_smote, y_smote= sm.fit_resample(X, y)
# x_train_smote, x_test_smote, y_train_smote, y_test_smote = train_test_split(x_smote, y_smote, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_smote,y_train_smote)

# y_pred_smote = hyper_SVC.predict(x_test_smote)
# print('\n\nEvaluation with SMOTE:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_smote, y_pred_smote) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote, y_pred_smote), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()



# ## ADASYN Random over-sampling
# ad = ADASYN(random_state=42)
# x_AD, y_AD = ad.fit_resample(X, y)
# x_train_adasyn, x_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(x_AD, y_AD, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_adasyn,y_train_adasyn)

# y_pred_adasyn = hyper_SVC.predict(x_test_adasyn)
# print('\n\nEvaluation with ADASYN:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_adasyn, y_pred_adasyn) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_adasyn, y_pred_adasyn), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # ## Random Under-Sampling 
# rus = RandomUnderSampler(random_state=42)
# x_rus, y_rus = rus.fit_resample(X, y)
# x_train_rus, x_test_rus, y_train_rus, y_test_rus = train_test_split(x_rus, y_rus, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_rus,y_train_rus)

# y_pred_rus = hyper_SVC.predict(x_test_rus)
# print('\n\nEvaluation with RandomUnderSampler:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_rus, y_pred_rus) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_rus, y_pred_rus), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # ## Near Miss Under-Sampling 
# nr = NearMiss()
# x_NM, y_NM = nr.fit_resample(X, y)
# x_train_nearMiss, x_test_nearMiss, y_train_nearMiss, y_test_nearMiss = train_test_split(x_NM, y_NM, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_nearMiss,y_train_nearMiss)

# y_pred_nearMiss = hyper_SVC.predict(x_test_nearMiss)
# print('\n\nEvaluation with Near Miss Under-Sampling:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_nearMiss, y_pred_nearMiss) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_nearMiss, y_pred_nearMiss), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn_SVM = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_smote_enn,y_train_smote_enn)

# y_pred_smote_enn_SVM = hyper_SVC.predict(x_test_smote_enn)
# print('\n\nEvaluation with SMOTEENN:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_smote_enn_SVM, y_pred_smote_enn_SVM) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_enn_SVM, y_pred_smote_enn_SVM), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# ## SMOTETomek
# smote_tomek = SMOTETomek(random_state=42)
# x_SK, y_SK = smote_tomek.fit_resample(X, y)
# x_train_smote_tomek , x_test_smote_tomek, y_train_smote_tomek, y_test_smote_tomek = train_test_split(x_SK, y_SK, test_size=0.25, random_state=42)

# hyper_SVC = SVC(C=10, kernel='poly', degree=3, gamma='scale') 
# hyper_SVC.fit(x_train_smote_tomek,y_train_smote_tomek)

# y_pred_smote_tomek = hyper_SVC.predict(x_test_smote_tomek)
# print('\n\nEvaluation with SMOTETomek:')
# print("Classification report of Hyper optimized SVM:\n", classification_report(y_test_smote_tomek, y_pred_smote_tomek) )

# # ## Karmaşıklık matrisi
# # print("Confussion matrix of Hyper optimized SVM:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test_smote_tomek, y_pred_smote_tomek), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()



# ## Tum modellere ait ROC eğrisi
# lr_false_positive_rate,lr_true_positive_rate,lr_threshold = roc_curve(y_test_smote_enn_LR,y_pred_smote_enn_LR)
# nb_false_positive_rate,nb_true_positive_rate,nb_threshold = roc_curve(y_test_smote_enn_NB,y_pred_smote_enn_NB)
# rf_false_positive_rate,rf_true_positive_rate,rf_threshold = roc_curve(y_test_smote_enn_RF,y_pred_smote_enn_RF)                                                             
# xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test_smote_enn_XGB,y_pred_smote_enn_XGB)
# knn_false_positive_rate,knn_true_positive_rate,knn_threshold = roc_curve(y_test_smote_enn_KNN,y_pred_smote_enn_KNN)
# dt_false_positive_rate,dt_true_positive_rate,dt_threshold = roc_curve(y_test_smote_enn_DT,y_pred_smote_enn_DT)
# svc_false_positive_rate,svc_true_positive_rate,svc_threshold = roc_curve(y_test_smote_enn_SVM,y_pred_smote_enn_SVM)


# sns.set_style('whitegrid')
# plt.figure(figsize=(10,5))
# plt.title('ROC Eğrisi')
# plt.plot(lr_false_positive_rate,lr_true_positive_rate,label='Logistic Regression')
# plt.plot(nb_false_positive_rate,nb_true_positive_rate,label='Naive Bayes')
# plt.plot(rf_false_positive_rate,rf_true_positive_rate,label='Random Forest')
# plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
# plt.plot(knn_false_positive_rate,knn_true_positive_rate,label='K-Nearest Neighbor')
# plt.plot(dt_false_positive_rate,dt_true_positive_rate,label='Desion Tree')
# plt.plot(svc_false_positive_rate,svc_true_positive_rate,label='Support Vector Classifier')
# plt.plot([0,1],ls='--')
# plt.plot([0,0],[1,0],c='.5')
# plt.plot([1,1],c='.5')
# plt.ylabel('True positive degeri')
# plt.xlabel('False positive degeri')
# plt.legend()
# plt.show()


# # # # Ensemble Learning Techniques with Evaluation ###################################################################################################

# # # ## BaggingClassifier - RF
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# # RF = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth= 10, max_features='auto', min_samples_leaf=2, min_samples_split=10) 
# # bc = BaggingClassifier(base_estimator = RF,random_state=42)
# # bc.fit(X_train, y_train)
# # y_pred=bc.predict(X_test)

# # print('\n\nEvaluation with BaggingClassifier:')
# # print("Classification report of Hyper optimized Random Forest:\n", classification_report(y_test, y_pred) )

# # # Karmaşıklık matrisi
# # print("Confussion matrix of Random Forest:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # ## BaggingClassifier - LR
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# # LR = LogisticRegression(C = 2, max_iter = 100, penalty = 'l2', solver = 'liblinear', class_weight = {0: 1, 1: 1})
# # bc = BaggingClassifier(base_estimator = LR,random_state=42)
# # bc.fit(X_train, y_train)
# # y_pred=bc.predict(X_test)

# # print('\n\nEvaluation with BaggingClassifier:')
# # print("Classification report of Hyper optimized LogisticRegression:\n", classification_report(y_test, y_pred) )

# # # Karmaşıklık matrisi
# # print("Confussion matrix of Logistic Regression:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()


# # # ## RUSBoostClassifier
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# # rusboost = RUSBoostClassifier(n_estimators=100, algorithm='SAMME.R',random_state=42)
# # rusboost.fit(X_train, y_train)
# # y_pred = rusboost.predict(X_test)

# # print('\n\nEvaluation with RUSBoostClassifier:')
# # print("Classification report of Hyper optimized model:\n", classification_report(y_test, y_pred) )

# # # Karmaşıklık matrisi
# # print("Confussion matrix of model:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()



# # # ## BalancedRandomForestClassifier
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# # brf = BalancedRandomForestClassifier( random_state=0, sampling_strategy="all", replacement=True)
# # brf.fit(X_train, y_train)
# # y_pred = brf.predict(X_test)

# # print('\n\nEvaluation with BalancedRandomForestClassifier:')
# # print("Classification report of Hyper optimized model:\n", classification_report(y_test, y_pred) )

# # # Karmaşıklık matrisi
# # print("Confussion matrix of model:")
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
# # plt.xlabel('Tahmin Edilen Sınıf')
# # plt.ylabel('Gerçek Sınıf')
# # plt.show()
