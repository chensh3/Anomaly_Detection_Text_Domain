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


SAMPLE_LOWER_LIMIT = 50
test_size = 0.3

attributes = pd.read_excel("attributes.xlsx")
attributes.rename(columns={"Unnamed: 0": "classes"}, inplace=True)
num_classes = 34
attributes.set_index("classes", inplace=True)

classes = pd.DataFrame(np.zeros((num_classes, num_classes)), index=attributes.index, columns=attributes.index,
                       dtype=int)
for i in range(num_classes):
    classes.iloc[i, i] = 1
attributes = pd.concat([classes, attributes], axis=1)



print("Loading data df...")
df = pd.read_pickle("short_df_all_192.pickle")
print("Done loading data")

permutations = list(itertools.combinations_with_replacement(attributes.columns, 2))
dataset_df = pd.DataFrame(
    columns=["Dataset_name", "Anomaly_classes", "Training_sample_size",
             "Test_sample_size",
             "Test_balance_size",
             "Test_length_dev",
             "Anomaly_All_ratio_dev",
             "Test_size_balance_dev",
             "Test_All_ratio_dev",
             "Test_Train_ratio_dev",
             "Anomaly_test_ratio_dev"])


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


def train_supervised(name,train, test):
    print("start train function")
    train_data=train.drop(columns=["short_description","category"])
    test_data=test.drop(columns=["short_description","category"])
    print("start setup")
    start = perf_counter()
    s = setup(train_data, test_data=test_data, target='label', session_id=42,index=False,fold=5,preprocess=False
              ,log_experiment = False,
              # experiment_name = name,
              fix_imbalance=True)
    print(f"setup time: {perf_counter()-start}")
        # clf1.setup(data=train.loc[:, ["embed","label"]], target='label',
    #            session_id=42,log_experiment = True, experiment_name = 'dataset')
    # start = perf_counter()
    # best_model = compare_models(include = ['lr'])
    best_model = compare_models()
    print(f"model train time: {perf_counter() - start}")
    results = pull()
    return results, best_model


count = 0
for i, per in tqdm(enumerate(permutations[:1])):

    # print("Working on per of: {per}")
    cls = np.array([])
    cls = np.append(cls, attributes[attributes[per[0]] == 1].index.values)
    cls = np.append(cls, attributes[attributes[per[1]] == 1].index.values)
    cls = list(set(cls))
    # print(f"Working on per of: {per} with classes of: {cls}")

    data = df.copy()

    anomaly = pd.DataFrame(columns=df.columns)
    for cls_1 in cls:
        anomaly = anomaly.append(data.loc[(data.category == cls_1), :])

    temp = anomaly.copy()  # For invert
    anomaly.loc[:, "label"] = np.ones(len(anomaly))
    normal = data.drop(anomaly.index)
    normal.loc[:, "label"] = np.zeros(len(normal))
    normal_train, normal_test = train_test_split(normal, test_size=test_size)
    anomaly_train, anomaly_test = train_test_split(anomaly, test_size=test_size)

    if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test, {per}")
        continue
    elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test, {per}")
        continue
    else:
        print("prepare dataset")
        count += 1
        name = f"{count}_{'_'.join(set(per))}"
        # dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
        # dataset_df.to_csv("dataset_df.csv")
        # test = pd.concat([anomaly_test, normal_test]).sample(frac=1).reset_index(drop=True)
        # sup_train = pd.concat([anomaly_train, normal_train]).sample(frac=1).reset_index(drop=True)
        test = pd.concat([anomaly_test, normal_test])
        sup_train = pd.concat([anomaly_train, normal_train])
        print("go to train models")
        results, best_model = train_supervised(name,sup_train, test)

    # INVERT:

    # # anomaly = normal.copy()
    # data = df.copy()
    # normal = temp.copy()
    # anomaly = data.drop(normal.index)
    #
    # normal.loc[:, "label"] = np.zeros(len(normal))
    # anomaly.loc[:, "label"] = np.ones(len(anomaly))
    #
    # normal_train, normal_test = train_test_split(normal, test_size=test_size)
    # anomaly_train, anomaly_test = train_test_split(anomaly, test_size=test_size)
    #
    # if len(anomaly_test) <= SAMPLE_LOWER_LIMIT:
    #     print(f"Not Enough Samples in Anomaly-Test,{i} {per}")
    #     continue
    # elif len(normal_test) <= SAMPLE_LOWER_LIMIT:
    #     print(f"Not Enough Samples in Normal-Test,{i} {per}")
    #     continue
    # else:
    #     count += 1
    #     name = f"{count}_{'_'.join(set(per))}_INVERT"
    #     cls = anomaly.category.unique()
    #     dataset_df = save_to_dataframe(dataset_df, name, cls, anomaly_test, normal_test, normal_train)
    #     dataset_df.to_csv("dataset_df.csv")
    #     test = pd.concat([anomaly_test, normal_test]).sample(frac=1).reset_index(drop=True)
    #     sup_train = pd.concat([anomaly_train, normal_train]).sample(frac=1).reset_index(drop=True)
    #     # results, best_model = train_supervised(name,sup_train, test)
