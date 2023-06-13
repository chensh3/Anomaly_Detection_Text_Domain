import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# results_data = pd.read_excel("results_initial_run.xlsx", skiprows=1, engine="openpyxl")
results_data = pd.read_csv("datasets_data_1.csv")
results_data.rename(columns={"Unnamed: 0":"dataset_name"},inplace=True)
results_data.vector_anomaly=results_data.vector_anomaly.astype(str)
results_data.vector_anomaly=results_data.vector_anomaly.str.zfill(9)

results_data.vector_normal=results_data.vector_normal.astype(str)
results_data.vector_normal=results_data.vector_normal.str.zfill(9)


results_data.set_index("dataset_name", inplace=True)

for ind in results_data.index:
    name = ind.split("_")

    diff = name[0]


    results_data.loc[ind, "hamm"] =diff

results_data.hamm = results_data.hamm.astype(int)

# for hamm in results_data.hamming_value.unique():
#     print(
#         f'\n Avg Results - hamming value of: {hamm}\n{results_data[results_data.hamming_value == hamm].loc[:, ["COPOD_ROC", "DeepSVDD_ROC", "COPOD_PR", "DeepSVDD_PR"]].apply(lambda x: [x.mean(), x.std()])}\n')
    # print(f'\n Avg Results:\n{results_data[results_data.hamming_value==hamm].mean()}\n')
# for name in tqdm(results_data.index):
#     data = np.load(os.path.join('D:/', 'anomaly_data', name + '.npz.npy'),
#                    allow_pickle = True).tolist()
#     # d = data["X_train"].to_numpy().reshape(data["X_train"].values.shape[0])
#     #
#     # d = data["X_test"].to_numpy().reshape(data["X_test"].values.shape[0])
#
#     # y_train = data["y_train"].values
#     y_test = data["y_test"].values
#     results_data.loc[name,["train_len","test_len","anomaly_len"]]=[len(data["X_train"]),len(data["X_test"]), y_test.sum()]
#     # print(data["X_train"].shape, data["X_test"].shape, y_test.sum(), y_train.sum())


# for name in results_data.index:
#     results_data.loc[name,"radius_diff"]=f"radius: {results_data.loc[name,'radius']} ; differance: {results_data.loc[name,'difficulty']}"


print(results_data.groupby("hamm").mean())


sns.lineplot(x=results_data.groupby("hamm").mean().index,y="COPOD_ROC",data=results_data.groupby("hamm").mean())
