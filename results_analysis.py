import numpy as np
import pandas as pd

# df = pd.read_excel("results_initial_run.xlsx", skiprows=1, engine="openpyxl")
df = pd.read_csv("results_initial_run.csv",skiprows=1)
df.drop(columns=["Unnamed: 3"],inplace=True)
# df = df.rename(
#     columns={"Unnamed: 0": "dataset_name", "COPOD": "COPOD_ROC", "DeepSVDD": "DeepSVDD_ROC", "COPOD.1": "COPOD_PR",
#              "DeepSVDD.1": "DeepSVDD_PR"})
df = df.rename(
    columns={"Unnamed: 0": "dataset_name", "COPOD": "COPOD_ROC", "DeepSVDD": "DeepSVDD_ROC", "COPOD.1": "COPOD_PR",
             "DeepSVDD.1": "DeepSVDD_PR"})
df.set_index("dataset_name", inplace=True)

for ind in df.index:
    name = ind.split("_")

    diff = name[0]
    hamm = name[1]
    vectors = [name[2], name[3]]
    classes = name[4:]

    df.loc[ind, ["difficulty", "hamming_value", "vectors", "classes"]] = [diff, hamm, vectors, classes]

df.hamming_value = df.hamming_value.astype(int)

for hamm in df.hamming_value.unique():
    print(
        f'\n Avg Results - hamming value of: {hamm}\n{df[df.hamming_value == hamm].loc[:, ["COPOD_ROC", "DeepSVDD_ROC", "COPOD_PR", "DeepSVDD_PR"]].apply(lambda x: [x.mean(), x.std()])}\n')
    # print(f'\n Avg Results:\n{df[df.hamming_value==hamm].mean()}\n')
