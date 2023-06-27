# scp .\short_df_all_192.pickle project33@192.168.90.203:~/chen/Anomaly_Detection_Text_Domain/short_df_all_192.pickle
# scp project33@192.168.90.203:~/chen/Anomaly_Detection_Text_Domain/full

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

# from data_generator import DataGenerator
from myutils import Utils
from baseline.PyOD import PYOD

SAMPLE_LOWER_LIMIT = 50
test_size = 0.3
CV_FOLD = 5
SEED = 42
attributes = pd.read_excel("attributes.xlsx")
attributes.rename(columns = {"Unnamed: 0": "classes"}, inplace = True)
num_classes = 34
attributes.set_index("classes", inplace = True)
np.random.seed(42)
classes = pd.DataFrame(np.zeros((num_classes, num_classes)), index = attributes.index, columns = attributes.index,
                       dtype = int)
for i in range(num_classes):
    classes.iloc[i, i] = 1
attributes = pd.concat([classes, attributes], axis = 1)

print("Loading data df...")
df = pd.read_pickle("short_df_all_192.pickle")
print("Done loading data")

permutations = list(itertools.combinations_with_replacement(attributes.columns, 2))

# datagenerator = DataGenerator()  # data generator
utils = Utils()  # utils function

model_dict = {'KNN': PYOD}
# model_dict = {'COPOD': PYOD, 'DeepSVDD': PYOD}  # WORKS
num_models = len(model_dict.keys())
ad_results = pd.DataFrame(
    columns = [["balance"] * 3 * num_models + ["non_balance"] * 3 * num_models, sorted(list(model_dict.keys()) * 3) * 2,
               ['acc', 'auc', 'f1'] * 2 * num_models])


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


def train_ad(train_data, test_data):
    train_data = train_data.drop(columns = ["short_description", "category"])
    test_data = test_data.drop(columns = ["short_description", "category"])

    test_balance = balance_test(test_data)

    dataset_result = []
    for test in [test_balance, test_data]:
        for j, (name, clf) in enumerate(model_dict.items()):
            print(f"\t#{j}/{len(model_dict)}    model: {name}")
            # model initialization
            model = clf(seed = SEED, model_name = name)

            # training, for unsupervised models the y label will be discarded
            # model = model.fit(X_train = train_data.drop(columns = "label").values, y_train = train_data[ "label"].values)
            model = model.fit(X_train = train_data.drop(columns = "label").values)


            # output predicted anomaly score on testing set

            score = model.predict_score(test.drop(columns = "label").values)
            # !! Transform continues score to binary classification with a threshold
            score = np.where(score >=model.model.threshold_, 1, 0)

            # evaluation
            result = utils.metric(y_true =  test["label"].values, y_score = score)
            dataset_result =dataset_result + [result['acc'], result['aucroc'], result['f1']]

    return dataset_result


count = 0
for i, per in tqdm(enumerate(permutations)):

    # print("Working on per of: {per}")
    cls = np.array([])
    cls = np.append(cls, attributes[attributes[per[0]] == 1].index.values)
    cls = np.append(cls, attributes[attributes[per[1]] == 1].index.values)
    cls = list(set(cls))
    # print(f"Working on per of: {per} with classes of: {cls}")

    data = df.copy()

    anomaly = pd.DataFrame(columns = df.columns)
    for cls_1 in cls:
        # anomaly = anomaly.append(data.loc[(data.category == cls_1), :])
        anomaly = pd.concat([anomaly, data.loc[(data.category == cls_1), :]])

    temp = anomaly.copy()  # For invert
    anomaly.loc[:, "label"] = np.ones(len(anomaly))
    normal = data.drop(anomaly.index)
    normal.loc[:, "label"] = np.zeros(len(normal))
    normal_train, normal_test = train_test_split(normal, test_size = test_size, random_state = 42)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size = test_size, random_state = 42)

    if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test, {per}")
        continue
    elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test, {per}")
        continue
    else:

        count += 1
        name = f"{count}_{'_'.join(set(per))}"
        print("\nprepare dataset ", name)
        # dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
        # dataset_df.to_csv("dataset_df.csv")

        test = pd.concat([anomaly_test, normal_test])

        results = train_ad(normal_train, test)
        print("start train ad models")
        ad_results.loc[len(ad_results)] = results
        ad_results.to_csv("ad_results.csv")

    # INVERT:

    # anomaly = normal.copy()
    data = df.copy()
    normal = temp.copy()
    anomaly = data.drop(normal.index)

    normal.loc[:, "label"] = np.zeros(len(normal))
    anomaly.loc[:, "label"] = np.ones(len(anomaly))

    normal_train, normal_test = train_test_split(normal, test_size = test_size, random_state = 42)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size = test_size, random_state = 42)

    if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test,{i} {per}")
        continue
    elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test,{i} {per}")
        continue
    else:
        count += 1
        name = f"{count}_{'_'.join(set(per))}_INVERT"
        print("\nprepare dataset ", name)
        # cls = anomaly.category.unique()
        # dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
        # dataset_df.to_csv("dataset_df.csv")
        test = pd.concat([anomaly_test, normal_test])

        results = train_ad(normal_train, test)
        ad_results.loc[len(ad_results)] = results
        ad_results.to_csv("ad_results.csv")
