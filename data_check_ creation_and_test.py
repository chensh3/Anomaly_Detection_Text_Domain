import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools
from tqdm.auto import tqdm

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

vector_to_class = {
    129: ["politics", "sports"],
    130: ["wellness", "parenting", "divorce"],
    4: ["entertainment", "comedy"],
    452: ["travel"],
    390: ["healthy living tips"],
    42: ["queer voices"],
    324: ["food & drink"],
    145: ["business", "crime"],
    40: ["black voices", "women"],
    198: ["home & living"],
    194: ["weddings"],
    258: ["impact"],
    257: ["world news", "u.s. news"],
    0: ["media", "religion"],
    1: ["good news", "weird news"],
    68: ["style"],
    280: ["science"],
    409: ["tech"],
    155: ["money"],
    264: ["environment"],
    128: ["fifty"],
    325: ["arts & culture"],
    136: ["college"],
    32: ["latino voices"],
    153: ["education"],
}

class_to_vector = {

    129: ["politics", "sports"],
    130: ["wellness", "parenting", "divorce"],
    4: ["entertainment", "comedy"],
    452: ["travel"],
    390: ["healthy living tips"],
    42: ["queer voices"],
    324: ["food & drink"],
    145: ["business", "crime"],
    40: ["black voices", "women"],
    198: ["home & living"],
    194: ["weddings"],
    258: ["impact"],
    257: ["world news", "u.s. news"],
    0: ["media", "religion"],
    1: ["good news", "weird news"],
    68: ["style"],
    280: ["science"],
    409: ["tech"],
    155: ["money"],
    264: ["environment"],
    128: ["fifty"],
    325: ["arts & culture"],
    136: ["college"],
    32: ["latino voices"],
    153: ["education"],

}
vectors = pd.DataFrame({"dec": vector_to_class.keys(), "classes": vector_to_class.values()})
vectors["bin"] = vectors.dec.apply(lambda x: np.binary_repr(x, 9))
bin2dec = lambda x: int(x, 2)

permutations = list(itertools.combinations_with_replacement(vectors.bin, 2))

results = [[], [], [], [], [], [], [], [], []]

hamm_mat = pd.DataFrame(np.zeros(shape = (len(vectors.bin.unique()), len(vectors.bin.unique()))),
                        index = vectors.bin.unique(), columns = vectors.bin.unique())
hamm_mat_classes = pd.DataFrame(
    np.zeros(shape = (np.hstack(vectors.classes).shape[0], np.hstack(vectors.classes).shape[0])),
    index = np.hstack(vectors.classes), columns = np.hstack(vectors.classes))

hamm_mat = hamm_mat.astype(int)
hamm_mat_classes = hamm_mat_classes.astype(int)

for per in permutations:
    hamm_value = int(hamming(list(str(per[0])), list(str(per[1]))) * 9)
    results[hamm_value].append(per)

    hamm_mat.loc[per[0], per[1]] = hamm_value
    hamm_mat.loc[per[1], per[0]] = hamm_value
    classes_0 = vectors[vectors.bin == per[0]].classes.to_list()[0]
    classes_1 = vectors[vectors.bin == per[1]].classes.to_list()[0]
    for cls_0 in classes_0:
        for cls_1 in classes_1:
            hamm_mat_classes.loc[cls_0, cls_1] = hamm_value
            hamm_mat_classes.loc[cls_1, cls_0] = hamm_value

## GET bin vector from class:
# vectors[vectors['classes'].astype(str).str.contains(cls_0)].bin


# # BERT
#
# from bert import bert_embed

# print(" start embed")
# embeding = bert_embed(df.short_description.to_list())
# print("stop embed")
# df["embed"] = embeding
# df.to_pickle("short_df_all.pickle")
df = pd.read_pickle("short_df_all.pickle")

# easy = results[0:2]
# medium = results[3:5]
# hard = results[7:]
dataset_df_columns = ["class_anomaly", "class_noraml", "vector_anomaly", "vector_normal", "len_anomaly", "len_test",
                      "len_train", "len_cls_anomaly", "cls_normal"]
datasets_data = pd.DataFrame(columns = dataset_df_columns)
# anomaly = pd.DataFrame(columns = df.columns)
count = 0
for i, cls_0 in enumerate(np.hstack(vectors.classes)):
    for cls_1 in tqdm(np.hstack(vectors.classes), desc = f" {cls_0}: {i + 1}/34"):
        hamm_value = hamm_mat_classes.loc[cls_0, cls_1]

        if len(df[df.category == cls_0]) >= len(df[df.category == cls_1]):
            cls_anomaly, cls_normal = cls_1, cls_0
        else:
            cls_anomaly, cls_normal = cls_0, cls_1
        anomaly = df[df.category == cls_anomaly].copy()
        normal = df[df.category == cls_normal].copy()
        anomaly.loc[:, "label"] = np.ones(len(anomaly))
        normal.loc[:, "label"] = np.zeros(len(normal))
        if 2 * len(anomaly) > 0.25 * len(normal):
            test_normal = normal.sample(int(len(normal) / 9))
            X_train = normal.drop(test_normal.index)
            anomaly = anomaly.sample(int(len(normal) / 9))
            X_test = anomaly.append(test_normal)
        else:
            test_normal = normal.sample(len(anomaly))
            X_test = anomaly.append(test_normal)
            X_train = normal.drop(test_normal.index)

        y_train = X_train.loc[:, ["label"]]
        y_test = X_test.loc[:, ["label"]]
        X_test = X_test.loc[:, ["embed"]]
        X_train = X_train.loc[:, ["embed"]]

        dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

        anomaly_vec = vectors[vectors['classes'].astype(str).str.contains(cls_anomaly)].bin.values[0]
        normal_vec = vectors[vectors['classes'].astype(str).str.contains(cls_normal)].bin.values[0]
        dataset_name=f"{hamm_value}_{anomaly_vec}_{cls_anomaly}_{normal_vec}_{cls_normal}"
        np.save(f"D:/anomaly_data_test/{dataset_name}.npz",dataset)
        datasets_data.loc[dataset_name,dataset_df_columns]=[cls_anomaly,cls_normal,anomaly_vec,normal_vec, y_test.sum()[0],len(X_test),len(X_train),len(df[df.category == cls_anomaly]),len(df[df.category == cls_normal])]
        count += 1
    datasets_data.to_csv("datasets_data.csv")

import os
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# import the necessary package
from data_generator import DataGenerator
from myutils import Utils

datagenerator = DataGenerator()  # data generator
utils = Utils()  # utils function

from baseline.PyOD import PYOD

# dataset and model list / dict
dataset_list = [x.split(".npz")[0] for x in os.listdir('D:/anomaly_data_test')]
dataset_list = sorted(dataset_list)

model_dict = {'COPOD': PYOD}
# , 'DeepSVDD': PYOD}  # WORKS

# save the results
df_AUCROC = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())
df_AUCPR = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())
dataset_data=pd.read_csv("datasets_data.csv")
# seed for reproducible results
seed = 42

for i, dataset in tqdm(enumerate(dataset_list)):
    '''
    la: ratio of labeled anomalies, from 0.0 to 1.0
    realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
    noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
    '''
    # if pd.isna(pd.read_pickle("results_aucpr.pickle").loc[dataset].COPOD):
    print("\n\n")
    print(f"#{i}/{len(dataset_list)} : Dataset Name: {dataset}")
    # import the dataset
    datagenerator.dataset = dataset  # specify the dataset name
    data = datagenerator.generator(la = -1, realistic_synthetic_mode = None,
                                   noise_type = None)  # la = -1 => Unsupervised training, with equal anomaly and normal data amount in test

    for j, (name, clf) in enumerate(model_dict.items()):
        print(f"\t#{j}/{len(model_dict)}    model: {name}")
        # model initialization
        clf = clf(seed = seed, model_name = name)

        # training, for unsupervised models the y label will be discarded
        clf = clf.fit(X_train = data['X_train'], y_train = data['y_train'])

        # output predicted anomaly score on testing set
        if name == "DAGMM":
            score = clf.predict_score(data["X_train"], data["X_test"])
        else:
            score = clf.predict_score(data['X_test'])

        # evaluation
        result = utils.metric(y_true = data['y_test'], y_score = score)
        print(f" results: \n\t AUCROC: \t {result['aucroc']:.4f} \n\t AUCPR: \t {result['aucpr']:.4f}")
        # save results
        df_AUCROC.loc[dataset, name] = result['aucroc']
        df_AUCPR.loc[dataset, name] = result['aucpr']
    df_AUCPR.to_pickle("results_aucpr.pickle")
    df_AUCROC.to_pickle("results_aucroc.pickle")
    # else:
    #     print("already done with:", dataset)
print(df_AUCROC)
print(df_AUCPR)
# df_AUCPR.to_pickle("results_aucpr.pickle")
# df_AUCROC.to_pickle("results_aucroc.pickle")
print(f"Avg score:\n\tAUCROC:\n{df_AUCROC.mean().to_string()}\n\tAUCPR:\n{df_AUCPR.mean().to_string()}")

#
#
#     # for i,type_data in enumerate([easy,medium,hard]):
# #     for diff in type_data:
# for i in [0, 1, 2, 3]:
#     for radius in [2, 3, 4, 5]:
#
#         dif = results[i]
#         for diff in tqdm(dif, desc = f"radius: {radius}, i {i}"):
#             data = df.copy()
#             anomaly = pd.DataFrame(columns = data.columns)
#
#             classes = vectors.loc[(vectors.bin == diff[0]) | (vectors.bin == diff[1]), "classes"].values
#             classes = np.hstack(classes)[0]
#
#             for class_1 in classes:
#                 anomaly = anomaly.append(data.loc[(data.category == class_1), :])
#             anomaly.loc[:, "label"] = np.ones(len(anomaly))
#             normal = data.drop(anomaly.index)
#             for vec in vectors.bin.unique():
#                 if vec not in diff:
#                     if not (hamm_mat.loc[vec, diff[0]] > radius and hamm_mat.loc[vec, diff[1]] > radius):
#                         classes_drop = vectors.loc[(vectors.bin == vec), "classes"].values
#                         classes_drop = np.hstack(classes_drop)[0]
#                         for cls_drop in classes_drop:
#                             # print("delete", cls_drop)
#                             normal = normal.drop(normal[normal.category == cls_drop].index)
#
#             normal.loc[:, "label"] = np.zeros(len(normal))
#             if len(normal) * 0.5 > len(anomaly):
#                 noraml_to_test = normal.sample(len(anomaly))
#                 # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)
#
#                 X_test = anomaly.append(noraml_to_test)
#                 X_train = normal.drop(noraml_to_test.index)
#             else:
#                 print(diff, classes, " no normal in test")
#                 continue
#                 X_test = anomaly
#                 X_train = normal
#             y_train = X_train.loc[:, ["label"]]
#             y_test = X_test.loc[:, ["label"]]
#             X_test = X_test.loc[:, ["embed"]]
#             X_train = X_train.loc[:, ["embed"]]
#
#             dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
#
#             # np.save(f"D:/anomaly_data/{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}.npz", dataset)
#             # diff_difficulty_radius_vec1_vec2_classes.npz
#             # print(f"{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}", len(X_train), len(X_test), y_test.sum())
#             # results_data.loc[f"{i}_{radius}_{diff[0]}_{diff[1]}_{'_'.join(classes)}",["train_len","test_len","anomaly_len"]]= len(X_train), len(X_test), y_test.sum()
#             #
#             # for name in results_data.index:
#             #     results_data.loc[
#             #         name, "radius_diff"] = f"radius: {results_data.loc[name, 'radius']} ; differance: {results_data.loc[name, 'difficulty']}"
