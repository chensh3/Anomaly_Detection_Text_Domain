import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

SAMPLE_LOWER_LIMIT = 50
test_size = 0.3

# indexs=input()
index = [3, 10, 2]

# attributes.to_csv("attributes.csv")


# permutations = list(itertools.combinations_with_replacement(vectors.bin, 2))


# # BERT
#
# from bert import bert_embed

# print(" start embed")
# embeding = bert_embed(df.short_description.to_list())
# print("stop embed")
# df["embed"] = embeding
# df.to_pickle("short_df_all.pickle")


print("Loading data df...")
df = pd.read_pickle("short_df_all.pickle")
print("Done loading data")

# count = 0
dataset_df = pd.read_csv("dataset_df.csv")

dataset_df_input  = pd.DataFrame(
    columns = ["Dataset_name", "Anomaly_classes", "Training_sample_size",
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

for i in tqdm(range(10)):
    row = dataset_df.iloc[i]
    count=row.Dataset_name.split("_")[0]

    inverted = "INVERT" in row.Dataset_name
    per = row.Dataset_name.split("_")[1:3]
    if "INVERT" in per:
        per=per[0]
    cls = eval(row.Anomaly_classes)

    data = df.copy()

    anomaly = pd.DataFrame(columns = df.columns)
    for cls_1 in cls:
        anomaly = anomaly.append(data.loc[(data.category == cls_1), :])

    if not inverted:
        print("Not inverted", row.Dataset_name)
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
            name = f"{count}_{'_'.join(set(per))}"
            test_length = len(anomaly_test) + len(normal_test)
            dataset_length = len(normal_train) + test_length
            indexs = np.argmin([len(anomaly_test), len(normal_test)])
            balance_type = "A" if indexs == 0 else "N"
            dataset_df_input = save_to_dataframe(dataset_df_input, name, cls, anomaly_test, normal_test, normal_train)

            dataset_df_input.to_csv("dataset_df_input.csv")
            test = pd.concat([anomaly_test, normal_test]).sample(frac = 1).reset_index(drop = True)
            sup_train = pd.concat([anomaly_train, normal_train]).sample(frac = 1).reset_index(drop = True)
            y_train = normal_train.loc[:, ["label"]]
            y_test = test.loc[:, ["label"]]
            X_test = test.loc[:, ["embed"]]
            X_train = normal_train.loc[:, ["embed"]]
            #
            dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    else:
        print("Inverted", row.Dataset_name)
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
            name = f"{count}_{'_'.join(set(per))}_INVERT"
            cls = anomaly.category.unique()
            test_length = len(anomaly_test) + len(normal_test)
            dataset_length = len(normal_train) + test_length
            indexs = np.argmin([len(anomaly_test), len(normal_test)])
            balance_type = "A" if indexs == 0 else "N"
            dataset_df_input = save_to_dataframe(dataset_df_input, name, cls, anomaly_test, normal_test, normal_train)
            dataset_df_input.to_csv("dataset_df_input.csv")
            test = pd.concat([anomaly_test, normal_test]).sample(frac = 1).reset_index(drop = True)
            sup_train = pd.concat([anomaly_train, normal_train]).sample(frac = 1).reset_index(drop = True)
            y_train = normal_train.loc[:, ["label"]]
            y_test = test.loc[:, ["label"]]
            X_test = test.loc[:, ["embed"]]
            X_train = normal_train.loc[:, ["embed"]]
            #
            dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
