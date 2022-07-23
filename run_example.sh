#!/bin/bash

pip3 install -q -r requirements.txt

python3 exploration.py \
	--parameter_fpath "data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv" \
	--ml_fpath "data/OA machine learning Jun. 15-22.xlsx - All Data.csv" \
	--linear_correlation_ouput_fpath "Results/linear_correlation_w_PF_at_77oC.xlsx"

python3 modeling.py \
	--parameter_fpath "data/Copy of Sample Parameter List 011222.xlsx - Sheet1.csv" \
	--ml_fpath "data/OA machine learning Jun. 15-22.xlsx - All Data.csv" \
	--linear_correlation_ouput_fpath "Results/linear_correlation_w_PF_at_77oC.xlsx" \
	--modeling "classification" \
	--w_cross_validation False \
	--w_synthetic_data False \
	--w_important_Features False \
	--w_grid_search False \
	--synthetic_data_size 100 \
	--train_test_split_ratio 0.3

regression_error_file="BaggingRegressor_samples_error.csv"
classification_error_file="AdaBoostClassifier_samples_error.csv"

echo ""

echo "Error results for Bagging Regressor:"
less "Results/Errors/${regression_error_file}" | head -10

echo ""

echo "Error results for AdaBoost Classifier:"
less "Results/Errors/${classification_error_file}" | head -10
