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
    129: [["politics", "sports"]],
    130: [["wellness", "parenting", "divorce"]],
    4: [["entertainment", "comedy"]],
    452: [["travel"]],
    390: [["healthy living tips"]],
    42: [["queer voices"]],
    324: [["food & drink"]],
    145: [["business", "crime"]],
    40: [["black voices", "women"]],
    198: [["home & living"]],
    194: [["weddings"]],
    258: [["impact"]],
    257: [["world news", "u.s. news"]],
    0: [["media", "religion"]],
    1: [["good news", "weird news"]],
    68: [["style"]],
    280: [["science"]],
    409: [["tech"]],
    155: [["money"]],
    264: [["environment"]],
    128: [["fifty"]],
    325: [["arts & culture"]],
    136: [["college"]],
    32: [["latino voices"]],
    153: [["education"]],
}

vectors = pd.DataFrame({"dec": vector_to_class.keys(), "classes": vector_to_class.values()})
vectors["bin"] = vectors.dec.apply(lambda x: np.binary_repr(x, 9))
bin2dec = lambda x: int(x, 2)

permutations = list(itertools.combinations_with_replacement(vectors.bin, 2))

results = [[], [], [], [], [], [], [], [], []]

hamm_mat = pd.DataFrame(np.zeros(shape = (len(vectors.bin.unique()), len(vectors.bin.unique()))),
                        index = vectors.bin.unique(), columns = vectors.bin.unique())

hamm_mat = hamm_mat.astype(int)

for per in permutations:
    results[int(hamming(list(str(per[0])), list(str(per[1]))) * 9)].append(per)
    hamm_mat.loc[per[0], per[1]] = int(hamming(list(str(per[0])), list(str(per[1]))) * 9)
    hamm_mat.loc[per[1], per[0]] = int(hamming(list(str(per[0])), list(str(per[1]))) * 9)

pairs = results[3]

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

anomaly = pd.DataFrame(columns = df.columns)

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