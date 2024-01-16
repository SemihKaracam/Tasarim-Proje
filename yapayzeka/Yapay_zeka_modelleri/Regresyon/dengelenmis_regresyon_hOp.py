

## Dengelenmiş veri seti ile hiper parametre optimizasyonu yapılan regresyon modelleri eğitildi.

## Dengeleme yöntemleri ve regresyon modellerinin performansları karşılaştırıldı.
## En iyi sonuca SMOTEEN yöntemi ve Rasgele Orman Regresyonu modeliyle ulasıldı.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder  #preprocessing
from sklearn.preprocessing import MinMaxScaler  #normalization
from sklearn.model_selection import train_test_split    #splitting data

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from imblearn.combine import SMOTEENN


#Duzenlenmis veri seti icin
data = pd.read_csv("data.csv")
df = data.copy()

# df.head()
# df.info()
# df.describe()

y = df.Heart_attack_risk
X = df.drop(["Heart_attack_risk"], axis =1).astype("float64")

#normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
 

# ##################### LİNEER REGRESSION #############################################################################################################
# ## Not: Lineer regresyon modelleri genellikle karmaşık hiperparametrelere sahip olmadığından, 
# ## cross-validation genellikle modelin performansını değerlendirmek için yeterli olacaktır.

# # SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# LR = LinearRegression()

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# LR.fit(x_train_smote_enn,y_train_smote_enn)

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-cross_val_score(LR, X, y, cv=kf, scoring=rmse_scorer).mean())


# # print('\n\nEvaluation with SMOTEENN and LinearRegression:')

# # print(f'Lineer Regresyon Modeli Katsayıları: {model.coef_}')
# # print(f'Lineer Regresyon Modeli Kesme Noktası: {model.intercept_}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# y_pred_smote_enn = LR.predict(x_test_smote_enn)
# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)

# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))


# # ##################### Polinominal Regresyon #############################################################################################################


# # SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Polinominal regresyon modeli için pipeline oluşturma
# model = make_pipeline(PolynomialFeatures(), LinearRegression())

# # Hiperparametre aralığını belirleme (örneğin, polinom derecesi için)
# param_grid = {'polynomialfeatures__degree': [1, 2, 3, 4]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn)

# # En iyi hiperparametreleri ve modeli alın
# best_degree = grid_search.best_params_['polynomialfeatures__degree']
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)

# print('\n\nEvaluation with SMOTEENN and Polinominal Regresyon:')

# print(f'En iyi polinom derecesi: {best_degree}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')


# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))




# #################### Ridge Regresyon #############################################################################################################


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Ridge regresyon modeli için pipeline oluşturma (veri ölçeklendirme eklenmiş)
# model = make_pipeline(StandardScaler(), Ridge())

# # Hiperparametre aralığını belirleme (örneğin, alpha için)
# param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn)

# # En iyi hiperparametreleri ve modeli alın
# best_alpha = grid_search.best_params_['ridge__alpha']
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)


# print('\n\nEvaluation with SMOTEENN and Ridge Regresyon:')

# print(f'En iyi alpha (ridge parametresi): {best_alpha}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')


# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ############################## Lasso Regresyon ##############################################################################


# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Lasso regresyon modeli için pipeline oluşturma (veri ölçeklendirme eklenmiş)
# model = make_pipeline(StandardScaler(), Lasso())

# # Hiperparametre aralığını belirleme (örneğin, alpha için)
# param_grid = {'lasso__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn)

# # En iyi hiperparametreleri ve modeli alın
# best_alpha = grid_search.best_params_['lasso__alpha']
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)


# print('\n\nEvaluation with SMOTEENN and Lasso Regresyon:')

# print(f'En iyi alpha (lasso parametresi): {best_alpha}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ####################################### Elastik Net Regresyon ########################################################################



# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Elastic Net regresyon modeli için pipeline oluşturma (veri ölçeklendirme eklenmiş)
# model = make_pipeline(StandardScaler(), ElasticNet())

# # Hiperparametre aralığını belirleme (örneğin, alpha ve l1_ratio için)
# param_grid = {'elasticnet__alpha': [0.001, 0.01, 0.1, 1, 10, 100],
#               'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(X, y)

# # En iyi hiperparametreleri ve modeli alın
# best_alpha = grid_search.best_params_['elasticnet__alpha']
# best_l1_ratio = grid_search.best_params_['elasticnet__l1_ratio']
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)

# print('\n\nEvaluation with SMOTEENN and Elastik Net Regresyon:')

# print(f'En iyi alpha (elastic net parametresi): {best_alpha}')
# print(f'En iyi l1_ratio (elastic net parametresi): {best_l1_ratio}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



# ######################################## Support Vector Regression (SVR) #######################################################################


# # SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)

# #SVR modeli için pipeline oluşturma (veri ölçeklendirme eklenmiş)
# model = make_pipeline(StandardScaler(), SVR())

# #Hiperparametre aralığını belirleme (örneğin, C ve epsilon için)
# param_grid = {'svr__C': [0.1, 1, 10],
#               'svr__epsilon': [0.01, 0.1, 0.2, 0.5]}

# #K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# #RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# #GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn.ravel())

# #En iyi hiperparametreleri ve modeli alın
# best_C = grid_search.best_params_['svr__C']
# best_epsilon = grid_search.best_params_['svr__epsilon']
# best_model = grid_search.best_estimator_

# #Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)


# print('\n\nEvaluation with SMOTEENN and Support Vector Regression (SVR):')
# #Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# print(f'En iyi C (SVR parametresi): {best_C}')
# print(f'En iyi epsilon (SVR parametresi): {best_epsilon}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# #Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# scaler = MinMaxScaler(feature_range=(0, 100))
# y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))




# ########################################## Decision Trees Regression #############################################################################################################



# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Decision Tree Regressor modeli için parametre aralığını belirleme
# param_grid = {'max_depth': [None, 5, 10, 15],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # Decision Tree Regressor modeli
# tree_model = DecisionTreeRegressor()

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(tree_model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn.ravel())

# # En iyi hiperparametreleri ve modeli alın
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)

# print('\n\nEvaluation with SMOTEENN and Decision Trees Regression:')

# print(f'En iyi hiperparametreler: {best_params}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))




############################# Random Forest Regression ####################################################################33


## SMOTEENN
smote_enn = SMOTEENN(random_state=42)
x_ST, y_ST = smote_enn.fit_resample(X, y)
x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# Random Forest Regressor modeli için parametre aralığını belirleme
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

# K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# RMSE hesaplama için özel bir skor fonksiyonu oluşturma
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# Random Forest Regressor modeli
forest_model = RandomForestRegressor()

# GridSearchCV kullanarak hiperparametre optimizasyonu
grid_search = GridSearchCV(forest_model, param_grid, cv=kf, scoring=rmse_scorer)
grid_search.fit(x_train_smote_enn, y_train_smote_enn.ravel())

# En iyi hiperparametreleri ve modeli alın
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Cross-validation ile RMSE hesaplama
cv_rmse = np.sqrt(-grid_search.best_score_)


print('\n\nEvaluation with SMOTEENN and Random Forest Regression:')

print(f'En iyi hiperparametreler: {best_params}')
print(f'Cross-Validation RMSE: {cv_rmse}')

# Test seti üzerinde tahminler yap
y_pred_smote_enn = best_model.predict(x_test_smote_enn)

mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# Sonuçları ekrana yazdır
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
scaler = MinMaxScaler(feature_range=(0, 100))
y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))


# İlk birkaç örnek çıktıyı kontrol et
print("Eğitim örnekleri için gerçek değerler:")
print(y_test_smote_enn[:10])

print("Tahmin edilen değerlerin yüzde olarak karşılığı:")
# print(y_pred[:5])
print(y_pred_scaled[:10])


# ############################# Gradient Boosting Regression ####################################################################33



# ## SMOTEENN
# smote_enn = SMOTEENN(random_state=42)
# x_ST, y_ST = smote_enn.fit_resample(X, y)
# x_train_smote_enn, x_test_smote_enn, y_train_smote_enn, y_test_smote_enn = train_test_split(x_ST, y_ST, test_size=0.25, random_state=42)


# # Gradient Boosting Regressor modeli için parametre aralığını belirleme
# param_grid = {'n_estimators': [50, 100, 150],
#               'learning_rate': [0.01, 0.1, 0.2],
#               'max_depth': [3, 5, 7],
#               'min_samples_split': [2, 5, 10],
#               'min_samples_leaf': [1, 2, 4]}

# # K-fold cross-validation için KFold nesnesi oluşturma (örneğin, 10 katlı cross-validation)
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # RMSE hesaplama için özel bir skor fonksiyonu oluşturma
# rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# # Gradient Boosting Regressor modeli
# gb_model = GradientBoostingRegressor()

# # GridSearchCV kullanarak hiperparametre optimizasyonu
# grid_search = GridSearchCV(gb_model, param_grid, cv=kf, scoring=rmse_scorer)
# grid_search.fit(x_train_smote_enn, y_train_smote_enn.ravel())

# # En iyi hiperparametreleri ve modeli alın
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# # Cross-validation ile RMSE hesaplama
# cv_rmse = np.sqrt(-grid_search.best_score_)

# print('\n\nEvaluation with SMOTEENN and Gradient Boosting Regression:') 

# print(f'En iyi hiperparametreler: {best_params}')
# print(f'Cross-Validation RMSE: {cv_rmse}')

# # Test seti üzerinde tahminler yap
# y_pred_smote_enn = best_model.predict(x_test_smote_enn)

# mse = mean_squared_error(y_test_smote_enn, y_pred_smote_enn)
# r2 = r2_score(y_test_smote_enn, y_pred_smote_enn)
# # Sonuçları ekrana yazdır
# print(f'Mean Squared Error (MSE): {mse:.2f}')
# print(f'R-squared (R2): {r2:.2f}')

# # # Modelin çıktılarını [0, 100] aralığına dönüştürmek için bir scaler kullan
# # scaler = MinMaxScaler(feature_range=(0, 100))
# # y_pred_scaled = scaler.fit_transform(y_pred_smote_enn.reshape(-1, 1))



