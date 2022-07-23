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

#%matplotlib inline
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Exploration

class Exploration(object):
  def __init__(self):
    pass

  def _read_data(self, parameter_fpath, ml_fpath):
    data_parameter = pd.read_csv(parameter_fpath)
    data_parameter = data_parameter.dropna(axis='columns')
    data_parameter = data_parameter.drop(['Double Shot?', 'Scan Statagy'], axis=1)

    data_ml = pd.read_csv(ml_fpath)

    # merge the two dataframes
    df_total = pd.merge(data_parameter, data_ml, how='outer', on = 'Sample')
    df_total=df_total.dropna().reset_index(drop=True)

    return data_parameter, data_ml, df_total

  def _plot_parameter_hist(self, data_parameter):
    plt.rcParams.update({'font.size': 18})
    attributes = ["Power (W)", "Speed (mm/s)", "Hatch (mm)",
                  "Layer (mm)", "Laser Focus (mm)"]

    data_parameter[attributes].hist(bins=50, figsize=(16,12))
    #plt.show()
    
    os.makedirs("plots", exist_ok=True)  # create the directory to store the confusion matrices
    plt.savefig("plots/parameter_histogram.png")   # save the confusion matrices to the file system
    

  def _plot_pearson_corr(self, data_parameter):
    plt.figure(figsize=(12,10))
    attributes = ["Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]
    cor = data_parameter[attributes].corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    #plt.show()
    
    os.makedirs("plots", exist_ok=True)  # create the directory to store the confusion matrices
    plt.savefig("plots/pearson_correlation.png")   # save the confusion matrices to the file system

  def _get_correlation_w_target(self, df_total, output_filename):
    attributes = ["Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]
    df_total[attributes].corrwith(df_total['PF at 77oC, mW/m K2'])
    
    #print(df_total[attributes].corrwith(df_total['PF at 77oC, mW/m K2']).round(4))

    df_total[attributes].corrwith(df_total['PF at 77oC, mW/m K2']).round(4).to_excel("Results/linear_correlation_w_PF_at_77oC.xlsx")

  # Statistical Correlation Plotting
  def _get_statistical_correlation(self, df_total):
    features = ["PF at 77oC, mW/m K2", "Power (W)", "Speed (mm/s)", "Hatch (mm)", "Layer (mm)", "Laser Focus (mm)"]
    _ = sns.pairplot(
        df_total[features], kind = 'reg', diag_kind = 'kde',
        plot_kws={'scatter_kws': {'alpha': 0.1}}
        )
        
    os.makedirs("plots", exist_ok=True)  # create the directory to store the confusion matrices
    plt.savefig("plots/kernel_density_plot.png")   # save the confusion matrices to the file system

# Driver Code
def main():
  # Data Exploration

  parser = argparse.ArgumentParser()

  parser.add_argument("-a", "--parameter_fpath", required=True, help="parameters", default="data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv")
  parser.add_argument("-b", "--ml_fpath", required=True, help="ml", default="data/OA machine learning Jun. 15-22.xlsx - All Data.csv")
  parser.add_argument("-c", "--linear_correlation_ouput_fpath", required=True, help="ouput_corr", default="Results/linear_correlation_w_PF_at_77oC.xlsx")


  args = parser.parse_args()
   
  parameter_fpath = args.parameter_fpath
  ml_fpath = args.ml_fpath
  output_filename = args.linear_correlation_ouput_fpath

  E = Exploration()  # instantiate exploration object

  '''
  parameter_fpath = "data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv"
  ml_fpath = "data/OA machine learning Jun. 15-22.xlsx - All Data.csv"
  output_filename = "Results/linear_correlation_w_PF_at_77oC.xlsx"
  '''

  data_parameter, data_ml, df_total = E._read_data(parameter_fpath, ml_fpath)

  E._plot_parameter_hist(data_parameter)

  E._plot_pearson_corr(data_parameter)

  E._get_correlation_w_target(df_total, output_filename)

  E._get_statistical_correlation(df_total)
  
  
if __name__ == "__main__":
  main()