#  jupyter notebook --ip=0.0.0.0
import pandas as pd
import numpy as np
# from scipy.spatial.distance import hamming
import itertools
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
# from pycaret.classification import ClassificationExperiment
# from pycaret.classification import *
# import pycaret
from time import perf_counter
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import lightgbm as lgb
from pyod.models.deep_svdd import DeepSVDD
# from data_generator import DataGenerator
from myutils import Utils
from baseline.PyOD import PYOD
import optuna

# datagenerator = DataGenerator()  # data generator
utils = Utils()  # utils function

model_dict = {'IForest': PYOD}
# model_dict = {'COPOD': PYOD, 'DeepSVDD': PYOD}  # WORKS
num_models = len(model_dict.keys())
ad_results = pd.DataFrame(
    columns = [["balance"] * 3 * num_models + ["non_balance"] * 3 * num_models, sorted(list(model_dict.keys()) * 3) * 2,
               ['acc', 'auc', 'f1'] * 2 * num_models])

train = pd.read_pickle("train.pickle")
test = pd.read_pickle("test.pickle")


def balance_test(test):
    len_anomalies = int(test.label.sum())
    len_normal = len(test) - len_anomalies
    min_ind = np.argmin([len_normal, len_anomalies])
    if min_ind == 1:
        normal = test[test.label == 0].sample(len_anomalies, random_state = 42)
        anomaly = test[test.label == 1]

    else:
        anomaly = test[test.label == 1].sample(len_normal, random_state = 42)
        normal = test[test.label == 0]
    assert len(normal) == len(anomaly) == [len_normal, len_anomalies][min_ind], \
        f"Wrong length: {len(normal), len(anomaly), [len_normal, len_anomalies][min_ind]}"
    balance_test = pd.concat([anomaly, normal])
    return balance_test


train = train.drop(columns = ["label", "short_description", "category"], errors = "ignore")
# y_test=test.label


b_test = balance_test(test)
y_test = b_test.label
b_test = b_test.drop(columns = ["label", "short_description", "category"], errors = "ignore")


# model = DeepSVDD(hidden_neurons =[4], preprocessing= False)
#
# m = model.fit(train)
#
# pred=m.decision_function(b_test)
#
# pred_b = np.where(pred >=m.threshold_, 1, 0)
#
# aucroc = roc_auc_score(y_true = y_test, y_score = pred_b)
# f1 = f1_score(y_test, pred_b)
# acc = accuracy_score(y_test, pred_b)
# print(f" model results: F1 : {f1} , AUC : {aucroc} , ACC : {acc} ")

def build_model(params):
    if params["n_layers"] == 1:
        hidden_neurons = [params["hidden_neurons1"]]
    elif params["n_layers"] == 2:
        hidden_neurons = [params["hidden_neurons1"], params["hidden_neurons2"]]
    elif params["n_layers"] == 3:
        hidden_neurons = [params["hidden_neurons1"], params["hidden_neurons2"], params["hidden_neurons3"]]
    elif params["n_layers"] == 4:
        hidden_neurons = [params["hidden_neurons1"], params["hidden_neurons2"], params["hidden_neurons3"],
                          params["hidden_neurons4"]]

    model = DeepSVDD(hidden_neurons = hidden_neurons,
                     preprocessing = params["preprocessing"],
                     contamination = params["contamination"],
                     dropout_rate = params["dropout_rate"]

                     )
    return model


def train_and_evaluate(model):
    m = model.fit(train)

    pred = m.decision_function(b_test)

    pred_b = np.where(pred >= m.threshold_, 1, 0)

    aucroc = roc_auc_score(y_true = y_test, y_score = pred_b)
    f1 = f1_score(y_test, pred_b)
    acc = accuracy_score(y_test, pred_b)
    print(f" model results: F1 : {f1} , AUC : {aucroc} , ACC : {acc} ")

    return f1

results=pd.DataFrame(columns=["f1",'n_layers','hidden_neurons1','hidden_neurons2','hidden_neurons3','hidden_neurons4','dropout_rate','contamination','preprocessing'])

def objective(trial):
    params = {
        'hidden_neurons1': trial.suggest_categorical("hidden_neurons1", [64, 32, 16, 8, 4, 2]),
        'hidden_neurons2': trial.suggest_categorical("hidden_neurons2", [64, 32, 16, 8, 4, 2]),
        'hidden_neurons3': trial.suggest_categorical("hidden_neurons3", [64, 32, 16, 8, 4, 2]),
        'hidden_neurons4': trial.suggest_categorical("hidden_neurons4", [64, 32, 16, 8, 4, 2]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.4),
        'contamination': trial.suggest_float('contamination', 0.01, 0.2),
        'n_layers': trial.suggest_int("n_unit", 1, 3),
        'preprocessing': trial.suggest_categorical("preprocessing", [True, False])
    }

    model = build_model(params)

    f1 = train_and_evaluate(model)
    results.loc[len(results)]=[f1,params["n_layers"],params["hidden_neurons1"],params["hidden_neurons2"],params["hidden_neurons3"],params["hidden_neurons4"],params["dropout_rate"],params["contamination"],params["preprocessing"]]
    results.to_csv("results_optimize_dsvdd.csv")
    return f1

EPOCHS = 100

study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials = 100)

# base results : model results: F1 : 0.17873739091012128 , AUC : 0.5221917202042747 , ACC : 0.7694522661830446


# [4,4,4]:  model results: F1 : 0.19811681849498583 , AUC : 0.5096432918266081 , ACC : 0.509643291826608

# [4] :  model results: F1 : 0.2167784729026909 , AUC : 0.5135755079112443 , ACC : 0.5135755079112443
# [4]  preprocess off :  model results: F1 : 0.2126624610473512 , AUC : 0.5150734949911057 , ACC : 0.5150734949911057
# hidden_neurons =[16], preprocessing= False  :  model results: F1 : 0.21331109763493022 , AUC : 0.514184065162438 , ACC : 0.514184065162438
