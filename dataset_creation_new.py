import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
SAMPLE_LOWER_LIMIT=50
test_size=0.3
df = pd.read_json("News_Category_Dataset_v3.json", lines = True)
df = df[["short_description", "category"]]
df.category = df.category.str.lower()

df.loc[(df.category == "the worldpost") | (df.category == "worldpost"), 'category'] = "world news"
df.loc[(df.category == "style & beauty"), 'category'] = "style"
df.loc[(df.category == "arts") | (df.category == "culture & arts"), 'category'] = "arts & culture"
df.loc[(df.category == "parents"), 'category'] = "parenting"
df.loc[(df.category == "taste"), 'category'] = "food & drink"
df.loc[(df.category == "green"), 'category'] = "environment"
df.loc[(df.category == "healthy living"), 'category'] = "healthy living tips"

attributes=pd.read_excel("attributes.xlsx")
attributes.rename(columns={"Unnamed: 0":"classes"},inplace=True)
num_classes=34
attributes.set_index("classes",inplace=True)

classes=pd.DataFrame(np.zeros((num_classes,num_classes)),index=attributes.index,columns=attributes.index,dtype=int)
for i in range(num_classes):
    classes.iloc[i,i]=1
attributes=pd.concat([classes,attributes],axis=1)
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
dataset_df=pd.DataFrame(columns=["Dataset_name","Anomaly_classes","Training_sample_size","Test_sample_size","Test_balance_size"])
df = pd.read_pickle("short_df_all.pickle")
permutations = list(itertools.combinations_with_replacement(attributes.columns, 2))
anomaly = pd.DataFrame(columns = df.columns)

for i,per in tqdm(enumerate(permutations)):
    print("Working on per of: {per}")
    cls=np.array([])
    cls=np.append(cls,attributes[attributes[per[0]]==1].index.values)
    cls=np.append(cls,attributes[attributes[per[1]]==1].index.values)
    cls=list(set(cls))
    print(f"Working on per of: {per} with classes of: {cls}")

    name=f"{i}_{per[0]}_{per[1]}"

    data = df.copy()


    for cls_1 in cls:
        anomaly = anomaly.append(data.loc[(data.category == cls_1), :])
    anomaly.loc[:, "label"] = np.ones(len(anomaly))
    normal = data.drop(anomaly.index)
    normal.loc[:, "label"] = np.zeros(len(normal))
    normal_train,normal_test=train_test_split(normal,test_size=test_size)
    anomaly_train,anomaly_test=train_test_split(anomaly,test_size=test_size)

    if len(anomaly_test)<=SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Anomaly-Test, {per}")
        continue
    elif len(normal_test)<=SAMPLE_LOWER_LIMIT:
        print(f"Not Enough Samples in Normal-Test, {per}")
        continue
    else:

        test_length=len(anomaly_test)+len(normal_test)
        dataset_length = len(normal_train)+test_length
        ind=np.argmin([len(anomaly_test),len(normal_test)])
        balance_type="A" if ind==0 else "N"
        dataset_df.iloc[-1]=[name,cls,len(normal_train),f"{test_length} ({test_length/dataset_length:.2f}%)",f"{min(len(anomaly_test),len(normal_test))} {balance_type}"] #
        dataset_df.to_csv("dataset_df.csv")


# for i,type_data in enumerate([easy,medium,hard]):
#     for diff in type_data:
for i in [0, 1, 2, 3]:
    for radius in [2, 3, 4, 5]:

        dif = results[i]
        for diff in tqdm(dif, desc = f"radius: {radius}, i {i}"):
            data = df.copy()
            anomaly = pd.DataFrame(columns = data.columns)

            classes = vectors.loc[(vectors.bin == diff[0]) | (vectors.bin == diff[1]), "classes"].values
            classes = np.hstack(classes)[0]

            for class_1 in classes:
                anomaly = anomaly.append(data.loc[(data.category == class_1), :])
            anomaly.loc[:, "label"] = np.ones(len(anomaly))
            normal = data.drop(anomaly.index)
            for vec in vectors.bin.unique():
                if vec not in diff:
                    if not (hamm_mat.loc[vec, diff[0]] > radius and hamm_mat.loc[vec, diff[1]] > radius):
                        classes_drop = vectors.loc[(vectors.bin == vec), "classes"].values
                        classes_drop = np.hstack(classes_drop)[0]
                        for cls_drop in classes_drop:
                            # print("delete", cls_drop)
                            normal = normal.drop(normal[normal.category == cls_drop].index)

            normal.loc[:, "label"] = np.zeros(len(normal))
            if len(normal) * 0.5 > len(anomaly):
                noraml_to_test = normal.sample(len(anomaly))
                # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)

                X_test = anomaly.append(noraml_to_test)
                X_train = normal.drop(noraml_to_test.index)
            else:
                print(diff, classes, " no normal in test")
                continue
                X_test = anomaly
                X_train = normal
            y_train = X_train.loc[:, ["label"]]
            y_test = X_test.loc[:, ["label"]]
            X_test = X_test.loc[:, ["embed"]]
            X_train = X_train.loc[:, ["embed"]]

            dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

            # np.save(f"D:/anomaly_data/{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}.npz", dataset)
            # diff_difficulty_radius_vec1_vec2_classes.npz
            # print(f"{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}", len(X_train), len(X_test), y_test.sum())
            # results_data.loc[f"{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}",["train_len","test_len","anomaly_len"]]= len(X_train), len(X_test), y_test.sum()
            #
            # for name in results_data.index:
            #     results_data.loc[
            #         name, "radius_diff"] = f"radius: {results_data.loc[name, 'radius']} ; differance: {results_data.loc[name, 'difficulty']}"