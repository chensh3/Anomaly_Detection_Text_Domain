import numpy as np
import pandas as pd

df=pd.read_excel("results_initial_run.xlsx",skiprows=1,engine="openpyxl")

df=df.rename(columns={"Unnamed: 0":"dataset_name","COPOD":"COPOD_ROC","DeepSVDD":"DeepSVDD_ROC","COPOD.1":"COPOD_PR","DeepSVDD.1":"DeepSVDD_PR"})

df.set_index("dataset_name",inplace=True)


for ind in df.index:
    name=ind.split("_")

    diff=name[0]
    hamm=name[1]
    vectors=[name[2],name[3]]
    classes=[]
    for cls in name[4:]:
        classes.append(cls)

    df.loc[ind,["difficulty","hamming_value","vectors","classes"]]=[diff,hamm,vectors,classes]


for hamm in df.hamming_value.unique():
    print(f"\n Avg Results:\n{df[df.hamming_value==hamm].mean()}\n")
