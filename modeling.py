#/bin/python3

import argparse

import pandas as pd
import numpy as np
import os
import copy
from pprint import pprint
from sklearn import metrics
from sklearn.metrics import (plot_confusion_matrix, mean_squared_error, r2_score,
                             ConfusionMatrixDisplay, accuracy_score, mean_absolute_error,
                             confusion_matrix, log_loss, classification_report,
                             roc_auc_score, roc_curve, precision_score, recall_score, f1_score
                             )
from sklearn.inspection import permutation_importance
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    KFold, RepeatedKFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (LogisticRegression, LinearRegression, SGDClassifier, ElasticNet)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    BaggingRegressor, 
    RandomForestClassifier, 
    BaggingClassifier, 
    AdaBoostClassifier,
    AdaBoostRegressor
)
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression
from sklearn.utils import resample
import math
from collections import defaultdict
import seaborn as sns
from sklearn.neighbors import KernelDensity
from mlxtend.evaluate import paired_ttest_5x2cv

import matplotlib.pyplot as plt

from exploration import Exploration as E

# Modeling
class Modeling(object):
  def __init__(self):
    pass

  def _get_competing_models_indices(self, modeling = "regression", w_cross_validation = False):
    best_model_1_idx, best_model_2_idx = 0, 0 
    if modeling == "regression" and not w_cross_validation:
      best_model_1_idx = 6
      best_model_2_idx = 7
    elif modeling == "regression" and w_cross_validation:
      best_model_1_idx = 0
      best_model_2_idx = 7
    
    elif modeling == "classification" and not w_cross_validation:
      best_model_1_idx = 4
      best_model_2_idx = 8
    elif modeling == "classification" and w_cross_validation:
      best_model_1_idx = 6
      best_model_2_idx = 9

    return best_model_1_idx, best_model_2_idx

  def _generate_synthetic_kde(self, train_data, synthetic_sample_size = 50):
    seed = 42
    rand_state = 42

    bandwidth_params = {'bandwidth': np.arange(0.01, 1, 0.05)}
    grid_search = GridSearchCV(KernelDensity(), bandwidth_params)

    grid_search.fit(train_data)
    kde = grid_search.best_estimator_

    new_samples = kde.sample(synthetic_sample_size, random_state=rand_state)

    cols = list(train_data.columns)
    df_synthetic_sample = pd.DataFrame(new_samples, columns = cols)


    return df_synthetic_sample


  # statistical significance testing on two best performing models
  def _stat_significance(self, model1, model2, X_test, y_test):
    t, p = paired_ttest_5x2cv(estimator1=model1, 
                              estimator2=model2, 
                              X=X_test, 
                              y=y_test, 
                              random_seed=1)

    # summarize
    print(f'The P-value is = {p:.3f}')
    print(f'The t-statistics is = {t:.3f}')


  def logloss(self, true_label, predicted, eps=1e-15):
    p = np.clip(predicted, eps, 1 - eps)
    if true_label == 1:
      return -np.log(p)
    else:
      return -np.log(1 - p)


  def _predict_individual_samples(self, modelName, trained_model, X_test_raw, X_test, y_test, modeling="regression", y_test_original = None):
    indices = X_test_raw.index

    dict_scores = {}  # key as column index and value as the mae

    for test_sample, y_test_label, idx in zip(X_test, y_test, list(indices)):
      y_pred = trained_model.predict(test_sample.reshape(1, -1))  # make predictions

      if modeling == "regression":
        error = mean_absolute_error([y_test_label], y_pred)
      else:    
        predicted = np.max(trained_model.predict_proba(test_sample.reshape(1, -1)))

        if y_test_label == 'Negative':
          y_test_label = 0
        else:
          y_test_label = 1

        error = self.logloss(y_test_label, predicted)

      dict_scores[idx] = error
    

    error_scores = pd.DataFrame.from_dict(dict_scores, orient='index')
    error_scores = error_scores.round(5)

    error_scores =  pd.merge(error_scores, X_test_raw, left_index=True, right_index=True)   # merge two dfs

    if modeling == "regression":
      error_scores.rename(columns={0:'MAE'}, inplace=True)
      error_scores = pd.merge(error_scores, y_test, left_index=True, right_index=True)

    elif modeling == "classification":
      error_scores.rename(columns={0:'Log Loss'}, inplace=True)
      error_scores = pd.merge(error_scores, y_test_original, left_index=True, right_index=True)

    # sort the dataframe by mae or log loss
    if modeling == "regression":
      error_scores = error_scores.sort_values(by=['MAE'])
    elif modeling == 'classification':
      error_scores = error_scores.sort_values(by=['Log Loss'])

    os.makedirs(f"Results/Errors", exist_ok=True)
    error_scores.to_excel(f"Results/Errors/{modelName}_samples_error.xlsx")

    X_test_raw.to_excel("Results/test_samples.xlsx")


  def _determine_feature_importance(self, modelName, trained_model, features, X_train, y_train):
    # for classification models
    if modelName in ["Naive Bayes", "Logistic Regression", "Random Forest", "BaggingClassifier",
                     "AdaBoostClassifier", "Decision Tree", "LinearSVM", "PolySVM", "RBFSVM", "MLP"]:

      perm_importance = permutation_importance(trained_model, X_train, y_train)

      features = np.array(features)

      sorted_idx = perm_importance.importances_mean.argsort()

      plt.figure()
      plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx], )
      plt.title(modelName)
      plt.xlabel("Permutation Importance")


      os.makedirs("feature_importance_plots", exist_ok=True)  # create the directory to store the plots
      plt.savefig(f"feature_importance_plots/{modelName}")   # save the plots to the file system


    
    if (modelName == "Linear Regression" or 
        modelName == "Lasso" or
        modelName == "Ridge" or
        modelName == "ElasticNet"
        ):

      perm_importance = permutation_importance(trained_model, X_train, y_train)

      features = np.array(features)

      sorted_idx = perm_importance.importances_mean.argsort()

      plt.figure()
      plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx], )
      plt.title(modelName)
      plt.xlabel("Permutation Importance")


      os.makedirs("feature_importance_plots", exist_ok=True)  # create the directory to store the plots
      plt.savefig(f"feature_importance_plots/{modelName}")   # save the plots to the file system

    
    if (modelName == "LinearSVR" or 
        modelName == "PolySVR" or 
        modelName == "RBFSVR"):
      
      perm_importance = permutation_importance(trained_model, X_train, y_train)

      features = np.array(features)

      sorted_idx = perm_importance.importances_mean.argsort()

      plt.figure()
      plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx], )
      plt.title(modelName)
      plt.xlabel("Permutation Importance")


      os.makedirs("feature_importance_plots", exist_ok=True)  # create the directory to store the plots
      plt.savefig(f"feature_importance_plots/{modelName}")   # save the plots to the file system

    if modelName == "DecisionTreeRegressor":
      perm_importance = permutation_importance(trained_model, X_train, y_train)


      features = np.array(features)

      sorted_idx = perm_importance.importances_mean.argsort()

      plt.figure()
      plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx], )
      plt.title(modelName)
      plt.xlabel("Permutation Importance")


      os.makedirs("feature_importance_plots", exist_ok=True)  # create the directory to store the plots
      plt.savefig(f"feature_importance_plots/{modelName}")   # save the plots to the file system

    if (modelName == "BaggingRegressor" or 
        modelName == "AdaBoostRegressor" or 
        modelName == "MLPRegressor"):
      

      perm_importance = permutation_importance(trained_model, X_train, y_train)

      features = np.array(features)

      sorted_idx = perm_importance.importances_mean.argsort()

      plt.figure()
      plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx], )
      plt.title(modelName)
      plt.xlabel("Permutation Importance")


      os.makedirs("feature_importance_plots", exist_ok=True)  # create the directory to store the plots
      plt.savefig(f"feature_importance_plots/{modelName}")   # save the plots to the file system


  def _get_seebeck_positive_negative_size(self, df):
    df_total = copy.deepcopy(df)
    target = "Seebeck at 77oC (350K) µV/K"
    df_total[df_total[target] < 0] = 0
    df_total[df_total[target] > 0] = 1

    df_total[target] = df_total[target].astype(str)

    df_total[df_total[target] == str(0.0)] = "Negative"
    df_total[df_total[target] == str(1.0)] = "Positive"

    print(pd.unique(df_total[target]))
    print(df_total[target].value_counts())

    print(df_total[target].value_counts().plot(kind='bar').set_title(target))

  # Method to prepare data for regression or classification task
  def _select_modeling(self, df_total, modeling="regression", w_important_features = False):
    df_total=df_total.dropna(axis=0)

    # Reset index after drop
    df_total=df_total.dropna().reset_index(drop=True)

    if w_important_features:
      features = ["Power (W)", "Hatch (mm)", "Laser Focus (mm)"]

    else:
      features = ["Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]
    X = df_total[features]

    if modeling == "regression":
      target = "PF at 77oC, mW/m K2"
      y = df_total[target]

    elif modeling == "classification":
      target = "Seebeck at 77oC (350K) µV/K"


      df_total_copy = df_total.copy(deep=True)
      y_original = df_total_copy[target]

      df_total[df_total[target] < 0] = 0
      df_total[df_total[target] > 0] = 1

      df_total[target] = df_total[target].astype(str)

      df_total[df_total[target] == str(0.0)] = "Positive"
      df_total[df_total[target] == str(1.0)] = "Negative"
      
      y = df_total[target].values

      return X, y, y_original

    return X, y

  # feature scaler
  def _scale_features(self, X):
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    return X

  def _select_and_fit_regression_model(self, modelName, X_train, y_train, 
                                       w_grid_search=False, w_target_interpolation=False):
    if modelName == 'Linear Regression':
      if not w_grid_search:
        model = LinearRegression()  # create object for the regression model
      else:
        model = LinearRegression()
        parameters = {
            "fit_intercept": [True, False],
            "normalize": [True, False], 
        }
        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    elif modelName == 'ElasticNet':
      if not w_grid_search:
        model = ElasticNet(random_state=0)
      else:
        model = ElasticNet(random_state=0)
        parameters = {
            "max_iter" : [1, 5, 10],
            "alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "l1_ratio" : np.arange(0.0, 1.0, 0.1)
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    
    elif modelName == 'Lasso':
      if not w_grid_search:
        model = linear_model.Lasso(alpha=0.1)
      else:
        model = linear_model.Lasso(alpha=0.1)
        parameters = {
            "max_iter" : [1, 5, 10],
            "alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    elif modelName == "Ridge":
      if not w_grid_search:
        model = linear_model.Ridge(alpha=1.0)
      else:
        model = linear_model.Ridge()

        parameters = {
            "max_iter" : [1, 5, 10],
            "alpha" : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            "solver" : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    elif modelName == "LinearSVR":
      if not w_grid_search:
        model = SVR(kernel = "linear", C=100, gamma="auto")
      else:
        model = SVR(kernel = "linear")

        parameters = {
            "C" : [1, 10, 100, 1000],
            "gamma" : ["auto", 0.0001, 0.001, 0.01]
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    elif modelName == "PolySVR":
      if not w_grid_search:
        model = SVR(kernel = "poly", C=100, gamma="auto")
      else:
        model = SVR(kernel = "poly")

        parameters = {
            "C" : [1, 10, 100, 1000],
            "gamma" : ["auto", 0.0001, 0.001, 0.01]
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid
    
    elif modelName == "RBFSVR":
      if not w_grid_search:
        model = SVR(kernel = "rbf", C=100, gamma="auto")
      else:
        model = SVR(kernel = "rbf")

        parameters = {
            "C" : [1, 10, 100, 1000],
            "gamma" : ["auto", 0.0001, 0.001, 0.01]
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid


    elif modelName == "DecisionTreeRegressor":
      if not w_grid_search:
        model = DecisionTreeRegressor(random_state=0)
      else:
        model = DecisionTreeRegressor(random_state=0)

        parameters ={
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None,1,2,3,4,5,6,7],
                    'max_features': [None, 'sqrt', 'auto', 'log2', 0.3,0.5,0.7],
                    'min_samples_split': [2,0.3,0.5,],
                    'min_samples_leaf':[1, 0.3,0.5]
                     }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid


    elif modelName == "BaggingRegressor":
      if not w_grid_search:
        #model = BaggingRegressor(base_estimator=SVR(), n_estimators = 10, random_state=0)
        model = BaggingRegressor(n_estimators = 10, random_state=0)
      else:
        model = BaggingRegressor()

        parameters = {
            'base_estimator': [None, SVR(), KNeighborsRegressor(), DecisionTreeRegressor()],
            'n_estimators': [10, 20, 50, 100],
            'max_samples': [0.5, 1.0, 5.0, 10.0, 20.0],
            'max_features': [0.5, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid

    elif modelName == "AdaBoostRegressor":
      if not w_grid_search:
        model = AdaBoostRegressor(n_estimators=10, random_state=0)
        #model = AdaBoostRegressor(base_estimator=LinearRegression(), random_state=0)
      else:
        model = AdaBoostRegressor()

        parameters = {
            'base_estimator': [None, LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor()],
            'n_estimators': [10, 20, 50, 100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 1, 10],
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid



    elif modelName == "MLPRegressor":
      if not w_grid_search:
        model = MLPRegressor(random_state=1, max_iter=10, hidden_layer_sizes=(100, 1))
      else:
        model = MLPRegressor( )

        parameters = {
            'hidden_layer_sizes': [(50, 1), (100, 1), (150, 1), (200, 1)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha' : [0.0001, 0.05],
            'learning_rate' : ['constant', 'adaptive']
        }

        grid = GridSearchCV(estimator=model, param_grid = parameters, scoring='r2', n_jobs = -1, cv = 2)

        grid.fit(X_train, y_train)

        return grid


    model.fit(X_train, y_train)  # fit the regression model

    return model

  def _evaluate_regression_model(self, model, X_test, y_test):
    y_pred = model.predict(X_test)  # make predictions

    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mse, rmse, mae
    

  def _regr_cross_validation(self, regr, X, y, cv=5):
    kf = KFold(n_splits=cv)
    #kf = StratifiedKFold(n_splits=cv, shuffle=True)

    lst_r2, lst_mse, lst_rmse, lst_mae = [], [], [], []
    
    for train_idx, test_idx in kf.split(X, y):
      X_train, X_test = np.array(X)[train_idx,:], np.array(X)[test_idx,:]
      y_train, y_test = y[train_idx], y[test_idx]

      regr.fit(X_train, y_train)

      regr, r2, mse, rmse, mae = self._evaluate_regression_model(regr, X_test, y_test)

      lst_r2.append(r2)
      lst_mse.append(mse)
      lst_rmse.append(rmse)
      lst_mae.append(mae)

    avg_r2 = np.asarray(lst_r2).mean()
    avg_mse = np.asarray(lst_mse).mean()
    avg_rmse = np.asarray(lst_rmse).mean()
    avg_mae = np.asarray(lst_mae).mean()

    return regr, avg_r2, avg_mse, avg_rmse, avg_mae


  # For seebeck +ve vs -ve classification
  def _select_classification_model(self, modelName):
    if modelName == "Naive Bayes":
      clf = GaussianNB()
    elif modelName == "Logistic Regression":
      clf = LogisticRegression()

    elif modelName == "Random Forest":
      clf = RandomForestClassifier(n_estimators=50, random_state=0)

    elif modelName == "BaggingClassifier":
      #clf = BaggingClassifier(base_estimator=SVC(), n_estimators=50, random_state=0)
      clf = BaggingClassifier(n_estimators=50, random_state=0)

    elif modelName == "AdaBoostClassifier":
      clf = AdaBoostClassifier(n_estimators=50, random_state=0)

    elif modelName == "Decision Tree":
      clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

    elif modelName == "LinearSVM":
      clf = SVC(kernel="linear", probability=True)

    elif modelName == "PolySVM":
      clf = SVC(kernel="poly", probability=True)

    elif modelName == "RBFSVM":
      clf = SVC(kernel="rbf", probability=True)

    elif modelName == "MLP":
      clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(200, ))

    return clf

  def _train(self, clf, X_train, y_train):
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    clf.fit(X_train, y_train)

    return clf


  def _clf_cross_validation(self, clf, X, y, cv=5):
    kf = StratifiedKFold(n_splits=cv, shuffle=True)

    lst_precision, lst_recall, lst_f1, lst_roc_auc, lst_test_log_loss = [], [], [], [], []

    for train_idx, test_idx in kf.split(X, y):
      X_train, X_test = np.array(X)[train_idx,:], np.array(X)[test_idx,:]
      y_train, y_test = y[train_idx], y[test_idx]

      clf.fit(X_train, y_train)

      
      precision, recall, f1_score, test_roc_auc_score, test_log_loss = self._predict(clf, X_test, y_test)

      lst_precision.append(precision)
      lst_recall.append(recall)
      lst_f1.append(f1_score)
      lst_roc_auc.append(test_roc_auc_score)
      lst_test_log_loss.append(test_log_loss)


    avg_precision = np.asarray(lst_precision).mean()
    avg_recall = np.asarray(lst_recall).mean()
    avg_f1_score = np.asarray(lst_f1).mean()
    avg_roc_auc = np.asarray(lst_roc_auc).mean()
    avg_log_loss = np.asarray(lst_test_log_loss).mean()

    return clf, avg_precision, avg_recall, avg_f1_score, avg_roc_auc, avg_log_loss

    

  def _feature_selection(self, clf):
    pass

  def _predict(self, clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    test_roc_auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


    test_log_loss = log_loss(y_test, clf.predict_proba(X_test))

    return precision, recall, f1_score, test_roc_auc_score, test_log_loss

  def _plot_confusion(self, clf, X_test, y_test, label, confusion_matrices_dir_name):
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot() 
    disp.ax_.set_title(label)

    os.makedirs(confusion_matrices_dir_name, exist_ok=True)  # create the directory to store the confusion matrices
    plt.savefig(f"{confusion_matrices_dir_name}/{label}")   # save the confusion matrices to the file system
    


  def _plot_roc_curve(self, clf, X_test, y_test, label):
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=f"{label}, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


  def _plot_roc_curve_all(self, lst_clf, X_test, y_test, lst_models):
    plt.figure(0).clf()
    
    for clf, label in zip(lst_clf, lst_models):
      y_pred_proba = clf.predict_proba(X_test)[::,1]
      fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
      auc = metrics.roc_auc_score(y_test, y_pred_proba)
      plt.plot(fpr,tpr,label=f"{label}, auc=" + str(auc))

    plt.legend(loc=4)
    plt.show()

      
# Driver Code
def main():
  # Modeling

  parser = argparse.ArgumentParser()

  parser.add_argument("-a", "--parameter_fpath", required=True, help="parameters", default="data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv")
  parser.add_argument("-b", "--ml_fpath", required=True, help="ml", default="data/OA machine learning Jun. 15-22.xlsx - All Data.csv")
  parser.add_argument("-c", "--linear_correlation_ouput_fpath", required=True, help="ouput_corr", default="Results/linear_correlation_w_PF_at_77oC.xlsx")

  parser.add_argument("-d", "--modeling", required=True, help="modeling", default="regression")
  parser.add_argument("-e", "--w_cross_validation", required=True, help="CV", default=False)
  parser.add_argument("-f", "--w_synthetic_data", required=True, help="synthetic data", default=False)
  parser.add_argument("-g", "--w_important_Features", required=True, help="important features", default=False)
  parser.add_argument("-s", "--w_grid_search", required=True, help="grid search", default=False)
  parser.add_argument("-i", "--synthetic_data_size", required=True, help="synthetic data size", default=100)
  parser.add_argument("-j", "--train_test_split_ratio", required=True, help="train test split ratio", default=0.3)


  args = parser.parse_args()
   
  parameter_fpath = args.parameter_fpath
  ml_fpath = args.ml_fpath
  output_filename = args.linear_correlation_ouput_fpath

  modeling = args.modeling
  w_cross_validation = eval(args.w_cross_validation)
  w_synthetic_data = eval(args.w_synthetic_data)
  
  w_important_Features = eval(args.w_important_Features)
  w_grid_search = eval(args.w_grid_search)
  synthetic_data_size = int(args.synthetic_data_size)
  train_test_split_ratio = float(args.train_test_split_ratio)



  M = Modeling() # Instantiate a modeling object

  '''
  modeling="classification"    # regression w.r.t power factor at 70 or type in classification w.r.t seebeck coefficient
  w_cross_validation = False  # applicable for both regression and classification tasks
  w_synthetic_data = False
  w_important_Features = False
  w_grid_search = False
  synthetic_data_size = 100
  train_test_split_ratio = 0.3
  
  parameter_fpath = "data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv"
  ml_fpath = "data/OA machine learning Jun. 15-22.xlsx - All Data.csv"
  output_filename = "Results/linear_correlation_w_PF_at_77oC.xlsx"
  '''

  _, _, df_total = E()._read_data(parameter_fpath, ml_fpath)


  if modeling == "regression":
    X, y = M._select_modeling(df_total, modeling, w_important_Features)
  elif modeling == "classification":
    X, y, y_original = M._select_modeling(df_total, modeling, w_important_Features)
  

  #X = M._scale_features(X) # scale features
  if modeling == "regression":
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, 
                                                        test_size=train_test_split_ratio, 
                                                        random_state=42)
  elif modeling == "classification":
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, 
                                                        test_size=train_test_split_ratio, 
                                                        random_state=42)
    
    _, _, _, y_test_original = train_test_split(X, y_original, 
                                                        test_size=train_test_split_ratio, 
                                                        random_state=42)

  

  if w_synthetic_data:
    train_data = pd.merge(X_train_raw, y_train, left_index=True, right_index=True)
    
    synthetic_data = M._generate_synthetic_kde(train_data, synthetic_data_size)

    X_synthetic = synthetic_data[list(X_train_raw.columns)]

    if modeling == "regression":
      y_synthetic = synthetic_data['PF at 77oC, mW/m K2']

    # To be completed
    elif modeling == "classification":
      y_synthetic = df_total["Seebeck at 77oC (350K) µV/K"]


    frames = [X_train_raw, X_synthetic]
    X_final_augmented = pd.concat(frames)

    y_final_augmented = pd.concat([y_train, y_synthetic], axis=0)


    X_train_raw = X_final_augmented
    y_train = y_final_augmented

  
  X_train = M._scale_features(X_train_raw) # scale features
  X_test = M._scale_features(X_test_raw) # scale features

  if modeling == "regression":
    lst_modelName = ["Linear Regression", 
                    "ElasticNet", 
                    "Lasso", 
                    "Ridge", 
                    "LinearSVR", 
                    "PolySVR", 
                    "BaggingRegressor",
                    "AdaBoostRegressor",
                    "MLPRegressor"]

    dict_scores = defaultdict(list)

    trained_models = []    # added on 0707
    for modelName in lst_modelName:
      model = M._select_and_fit_regression_model(modelName, X_train, y_train, w_grid_search)   # fits a regression model


      M._predict_individual_samples(modelName, model, X_test_raw, X_test, y_test)  # compute MAE for each test/train sample

      
      if not w_cross_validation:
        if w_important_Features:
          features = ["Power (W)", "Hatch (mm)", "Laser Focus (mm)"]
        else:
          features = ["Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]  # parameters
        M._determine_feature_importance(modelName, model, features, X_test, y_test)   # call to feature importance method
        output_filename = "regression_wrt_pf_at_70_results_wo_KFold.xlsx"

        clf, r2, mse, rmse, mae = M._evaluate_regression_model(model, X_test, y_test)
        trained_models.append(clf)

      elif w_cross_validation:
        output_filename = "regression_wrt_pf_at_70_results_w_KFold.xlsx"

        clf, r2, mse, rmse, mae = M._regr_cross_validation(model, X, y, 5)
        trained_models.append(clf)

      dict_scores["Regression Model"].append(modelName)
      dict_scores["R2"].append(r2)
      dict_scores["MSE"].append(mse)
      dict_scores["RMSE"].append(rmse)
      dict_scores["MAE"].append(mae)
      
    df_scores = pd.DataFrame(dict_scores)

    # write the dataframe to the file system where results are saved in
    os.makedirs("Results", exist_ok=True)


    df_scores = df_scores.round(4)
    df_scores.to_excel(f"Results/{output_filename}")

  elif modeling == "classification" :
    confusion_matrices_dir_name = "confusion_matrices"

    M._get_seebeck_positive_negative_size(df_total)

    lst_modelName = ["Naive Bayes", 
                    "Logistic Regression", 
                    "Random Forest",
                    "BaggingClassifier",
                    "AdaBoostClassifier",
                    "Decision Tree",
                    "LinearSVM",
                    "PolySVM",
                    "RBFSVM",
                    "MLP"]
    
    dict_scores = defaultdict(list)

    trained_models = []    # added on 0707
    for modelName in lst_modelName:
      # select model
      clf = M._select_classification_model(modelName)

      if w_cross_validation:
        output_filename = "classification_wrt_Seebeck_results_w_KFold.xlsx"
        clf, precision, recall, f1_score, test_roc_auc_score, test_log_loss = M._clf_cross_validation(clf, X, y, 5)

        trained_models.append(clf)    # added on 0707

      else:
        output_filename = "classification_wrt_Seebeck_results_wo_KFold.xlsx"
        # train model w/o KFold Cross-validation
        clf = M._train(clf, X_train, y_train)
        trained_models.append(clf)    # added on 0707

        M._predict_individual_samples(modelName, clf, X_test_raw, X_test, y_test, modeling, y_test_original)  # compute log_loss for each test/train sample

        if w_important_Features:
          features = ["Power (W)", "Hatch (mm)", "Laser Focus (mm)"]
        else:
          features = ["Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]  # parameters


        M._determine_feature_importance(modelName, clf, features, X_test, y_test)   # call to feature importance method

        # evaluate model
        precision, recall, f1_score, test_roc_auc_score, test_log_loss = M._predict(clf, X_test, y_test)


      dict_scores["Classification Model"].append(modelName)
      dict_scores["Precision"].append(precision)
      dict_scores["Recall"].append(recall)
      dict_scores["F1 Score"].append(f1_score)
      dict_scores["roc_auc_score"].append(test_roc_auc_score)
      dict_scores["test_log_loss"].append(test_log_loss)

      M._plot_confusion(clf, X_test, y_test, modelName, confusion_matrices_dir_name)
      
    df_scores = pd.DataFrame(dict_scores)

    # write the dataframe to the file system where results are saved in
    os.makedirs("Results", exist_ok=True)


    df_scores = df_scores.round(4)
    df_scores.to_excel(f"Results/{output_filename}")

  # To be improved further - Call the method to return competing models - statistical significance - get best model indices
  best_model_1_idx, best_model_2_idx = M._get_competing_models_indices(modeling, w_cross_validation)

  best_model_1 = trained_models[best_model_1_idx]
  best_model_2 = trained_models[best_model_2_idx]

  M._stat_significance(best_model_1, best_model_2, X_train, y_train)


if __name__ == "__main__":
  main()
