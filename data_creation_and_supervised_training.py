import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment
from pycaret.classification import *
import pycaret
from time import perf_counter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegressionCV

SAMPLE_LOWER_LIMIT = 50
test_size = 0.3
CV_FOLD = 5
SEED = 42
attributes = pd.read_excel("attributes.xlsx")
attributes.rename(columns = {"Unnamed: 0": "classes"}, inplace = True)
num_classes = 34
attributes.set_index("classes", inplace = True)

classes = pd.DataFrame(np.zeros((num_classes, num_classes)), index = attributes.index, columns = attributes.index,
                       dtype = int)
for i in range(num_classes):
    classes.iloc[i, i] = 1
attributes = pd.concat([classes, attributes], axis = 1)

print("Loading data df...")
df = pd.read_pickle("short_df_all_192.pickle")
print("Done loading data")

permutations = list(itertools.combinations_with_replacement(attributes.columns, 2))
dataset_df = pd.DataFrame(
    columns = ["Dataset_name", "Anomaly_classes", "Training_sample_size",
               "Test_sample_size",
               "Test_balance_size",
               "Test_length_dev",
               "Anomaly_All_ratio_dev",
               "Test_size_balance_dev",
               "Test_All_ratio_dev",
               "Test_Train_ratio_dev",
               "Anomaly_test_ratio_dev"])
non_balance_results = pd.DataFrame(
    columns = ["Dataset_name",
               "acc_avg",
               "acc_std",
               "acc_min",
               "acc_max",
               "acc_min_name",
               "acc_max_name",
               "auc_avg",
               "auc_std",
               "auc_min",
               "auc_max",
               "auc_min_name",
               "auc_max_name",
               "f1_avg",
               "f1_std",
               "f1_min",
               "f1_max",
               "f1_min_name",
               "f1_max_name",

               ])
balance_results = pd.DataFrame(
    columns = ["Dataset_name",
               "acc_avg",
               "acc_std",
               "acc_min",
               "acc_max",
               "acc_min_name",
               "acc_max_name",
               "auc_avg",
               "auc_std",
               "auc_min",
               "auc_max",
               "auc_min_name",
               "auc_max_name",
               "f1_avg",
               "f1_std",
               "f1_min",
               "f1_max",
               "f1_min_name",
               "f1_max_name",

               ])


def save_to_dataframe(dataset_data, db_name, cls_list, a_test, n_test, n_train):
    test_length = len(a_test) + len(n_test)
    dataset_length = len(n_train) + test_length
    ind = np.argmin([len(a_test), len(n_test)])
    balance_type = "A" if ind == 0 else "N"
    dataset_data.loc[len(dataset_data), :] = [db_name, cls_list, len(n_train),
                                              f"{test_length} ({len(a_test) / test_length * 100:.1f}%)",
                                              f"{min(len(a_test), len(n_test))} {balance_type}",
                                              test_length,  # Test_length_dev
                                              len(a_test) / dataset_length * 100,  # Anomaly_All_ratio_dev
                                              min(len(a_test), len(n_test)),  # Test_size_balance_dev
                                              test_length / dataset_length * 100,  # Test_All_ratio_dev
                                              test_length / len(n_train) * 100,  # Test_Train_ratio_dev
                                              len(a_test) / test_length * 100  # Anomaly_test_ratio_dev
                                              ]

    return dataset_data


def balance_test(test):
    len_anomalies = int(test.label.sum())
    len_normal = len(test) - len_anomalies
    min_ind = np.argmin([len_normal, len_anomalies])
    if min_ind == 1:
        normal = test[test.label == 0].sample(len_anomalies)
        anomaly = test[test.label == 1]

    else:
        anomaly = test[test.label == 1].sample(len_normal)
        normal = test[test.label == 0]
    assert len(normal) == len(anomaly) == [len_normal, len_anomalies][min_ind], \
        f"Wrong length: {len(normal), len(anomaly), [len_normal, len_anomalies][min_ind]}"
    balance_test = pd.concat([anomaly, normal])
    return balance_test


def balance_train(data):
    len_anomalies = int(data.label.sum())
    len_normal = len(data) - len_anomalies
    min_ind = np.argmin([len_normal, len_anomalies])
    if [len_normal, len_anomalies][min_ind] < 1000:
        return None
    elif [len_normal, len_anomalies][min_ind] > 10000:
        a = data[data.label == int(not min_ind)].sample([len_normal, len_anomalies][min_ind])
        b = data[data.label == min_ind]
        balance_train = pd.concat([a, b])
    else:
        if min_ind == 1:
            oversample = SMOTE()
            if len_normal / len_anomalies <= 1.3:
                balance_train, y = oversample.fit_resample(data.drop(columns = "label"), data.label)
                balance_train["label"] = y
            else:
                normal = data[data.label == 0].sample(int(len_anomalies * 1.3))
                anomaly = data[data.label == 1]
                balance_train = pd.concat([anomaly, normal])
                balance_train, y = oversample.fit_resample(balance_train.drop(columns = "label"), balance_train.label)
                balance_train["label"] = y
        else:
            oversample = SMOTE()
            if len_anomalies / len_normal <= 1.3:
                balance_train, y = oversample.fit_resample(data.drop(columns = "label"), data.label)
                balance_train["label"] = y
            else:
                anomaly = data[data.label == 1].sample(int(len_normal * 1.3))
                normal = data[data.label == 0]
                balance_train = pd.concat([anomaly, normal])
                balance_train, y = oversample.fit_resample(balance_train.drop(columns = "label"), balance_train.label)
                balance_train["label"] = y

    return balance_train


def get_info_from_results(results):
    info = np.array([])
    for col in ["Accuracy", "AUC", "F1"]:
        info = np.append(info, [results[col].mean(),
                                results[col].std(),
                                results[col].min(),
                                results[col].max(),
                                results.index[results[col].argmin()],
                                results.index[results[col].argmax()]]
                         )

    return info


def model_train(train, test):
    print("start setup")
    start = perf_counter()
    s = setup(train, test_data = test, target = 'label', session_id = SEED, index = False,
              n_jobs = -1,
              preprocess = False,
              log_experiment = False,
              use_gpu = True,
              # experiment_name = name,
              fix_imbalance = False)
    print(f"setup time: {perf_counter() - start}")
    best_model = compare_models(cross_validation=False)
    print(f"model train time: {perf_counter() - start}")
    results = pull()
    return results, best_model


def train_supervised(name, train, test):
    print("start train function")
    train_data = train.drop(columns = ["short_description", "category"])
    test_data = test.drop(columns = ["short_description", "category"])
    balance_train_df = balance_train(train_data)

    non_balance_results, _ = model_train(balance_train_df, test_data)
    balance_test_df = balance_test(test_data)

    balance_results, _ = model_train(balance_train_df, balance_test_df)

    non_balance_results.to_csv(f"full_test_results/non_balance_{name}.csv")
    balance_results.to_csv(f"full_test_results/balance_{name}.csv")

    balance_info = get_info_from_results(balance_results)
    non_balance_info = get_info_from_results(non_balance_results)

    return non_balance_info, balance_info


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
        anomaly = anomaly.append(data.loc[(data.category == cls_1), :])

    temp = anomaly.copy()  # For invert
    anomaly.loc[:, "label"] = np.ones(len(anomaly))
    normal = data.drop(anomaly.index)
    normal.loc[:, "label"] = np.zeros(len(normal))
    normal_train, normal_test = train_test_split(normal, test_size = test_size)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size = test_size)

    if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test, {per}")
        continue
    elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test, {per}")
        continue
    else:

        count += 1
        name = f"{count}_{'_'.join(set(per))}"
        print("prepare dataset ", name)
        # dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
        # dataset_df.to_csv("dataset_df.csv")
        # test = pd.concat([anomaly_test, normal_test]).sample(frac=1).reset_index(drop=True)
        # sup_train = pd.concat([anomaly_train, normal_train]).sample(frac=1).reset_index(drop=True)
        test = pd.concat([anomaly_test, normal_test])
        sup_train = pd.concat([anomaly_train, normal_train])
        print("go to train models")
        non_balance, balance = train_supervised(name,sup_train, test)
        non_balance_results.loc[len(non_balance_results), :] = np.append(name, non_balance)
        balance_results.loc[len(balance_results), :] = np.append(name, balance)
        non_balance_results.to_csv("non_balance_results.csv")
        balance_results.to_csv("balance_results.csv")

    # INVERT:

    # anomaly = normal.copy()
    data = df.copy()
    normal = temp.copy()
    anomaly = data.drop(normal.index)

    normal.loc[:, "label"] = np.zeros(len(normal))
    anomaly.loc[:, "label"] = np.ones(len(anomaly))

    normal_train, normal_test = train_test_split(normal, test_size = test_size)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size = test_size)

    if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test,{i} {per}")
        continue
    elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test,{i} {per}")
        continue
    else:
        count += 1
        name = f"{count}_{'_'.join(set(per))}_INVERT"
        print("prepare dataset ", name)
        # cls = anomaly.category.unique()
        # dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
        # dataset_df.to_csv("dataset_df.csv")
        test = pd.concat([anomaly_test, normal_test])
        sup_train = pd.concat([anomaly_train, normal_train])
        print("go to train models")
        non_balance, balance = train_supervised(name,sup_train, test)
        non_balance_results.loc[len(non_balance_results), :] = np.append(name, non_balance)
        balance_results.loc[len(balance_results), :] = np.append(name, balance)
        non_balance_results.to_csv("non_balance_results.csv")
        balance_results.to_csv("balance_results.csv")
