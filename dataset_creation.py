import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools

df = pd.read_json("News_Category_Dataset_v3.json", lines = True)
df = df[["headline", "category"]]
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

for per in permutations:
    results[int(hamming(list(str(per[0])), list(str(per[1]))) * 9)].append(per)

#BERT

# from bert import bert_embed
# print(" start embed")
# embeding=bert_embed(df.headline.to_list())
# print("stop embed")
# df["embed"]=embeding


easy = results[0:2]
medium = results[3:5]
hard = results[7:]

anomaly=pd.DataFrame(columns=df.columns)

# for i,type_data in enumerate([easy,medium,hard]):
#     for diff in type_data:

for i, diff in enumerate(easy):
    data=df.copy()
    anomaly = pd.DataFrame(columns=data.columns)
    if diff[1][0]!=diff[1][1]:
        classes=vectors.loc[(vectors.bin==diff[1][0]) | (vectors.bin==diff[1][1]),"classes" ].values
    else:
        classes=vectors.loc[(vectors.bin==diff[1][0]),"classes" ].values

        classes=np.hstack(classes)[0]
        for class_1 in classes :

            anomaly = anomaly.append(data.loc[(data.category == class_1), :])
        anomaly.loc[:,"label"] =np.ones(len(anomaly))
        normal = data.drop(anomaly.index)
        normal.loc[:,"label"]  = np.zeros(len(normal))
        noraml_to_test=normal.sample(len(anomaly))
        # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)

        X_test = anomaly.append(noraml_to_test)
        X_train = normal.drop(noraml_to_test.index)
        y_train = X_train.loc[:,["label"]]
        y_test = X_test.loc[:,["label"]]
        X_test = X_test.loc[:,["embed"]]
        X_train = X_train.loc[:,["embed"]]

        dataset={'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
        np.save(f"news_data/easy_{i}_{diff[0]}_{diff[1]}_{'_'.join(classes)}.npz", dataset)

for i,diff in enumerate(medium):
    data = df.copy()
    anomaly = pd.DataFrame(columns=data.columns)
    if diff[1][0] != diff[1][1]:
        classes = vectors.loc[(vectors.bin == diff[1][0]) | (vectors.bin == diff[1][1]), "classes"].values
    else:
        classes = vectors.loc[(vectors.bin == diff[1][0]), "classes"].values

        classes = np.hstack(classes)[0]
        for class_1 in classes:
            anomaly = anomaly.append(data.loc[(data.category == class_1), :])
        anomaly.loc[:, "label"] = np.ones(len(anomaly))
        normal = data.drop(anomaly.index)
        normal.loc[:, "label"] = np.zeros(len(normal))
        noraml_to_test = normal.sample(len(anomaly))
        # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)

        X_test = anomaly.append(noraml_to_test)
        X_train = normal.drop(noraml_to_test.index)
        y_train = X_train.loc[:, ["label"]]
        y_test = X_test.loc[:, ["label"]]
        X_test = X_test.loc[:, ["embed"]]
        X_train = X_train.loc[:, ["embed"]]

        dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        np.save(f"news_data/medium_{i}_{diff[0]}_{diff[1]}_{'_'.join(classes)}.npz", dataset)

for i,diff in enumerate(hard):
    data = df.copy()
    anomaly = pd.DataFrame(columns=data.columns)
    if diff[1][0] != diff[1][1]:
        classes = vectors.loc[(vectors.bin == diff[1][0]) | (vectors.bin == diff[1][1]), "classes"].values
    else:
        classes = vectors.loc[(vectors.bin == diff[1][0]), "classes"].values

        classes = np.hstack(classes)[0]
        for class_1 in classes:
            anomaly = anomaly.append(data.loc[(data.category == class_1), :])
        anomaly.loc[:, "label"] = np.ones(len(anomaly))
        normal = data.drop(anomaly.index)
        normal.loc[:, "label"] = np.zeros(len(normal))
        noraml_to_test = normal.sample(len(anomaly))
        # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)

        X_test = anomaly.append(noraml_to_test)
        X_train = normal.drop(noraml_to_test.index)
        y_train = X_train.loc[:, ["label"]]
        y_test = X_test.loc[:, ["label"]]
        X_test = X_test.loc[:, ["embed"]]
        X_train = X_train.loc[:, ["embed"]]

        dataset = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        np.save(f"news_data/hard_{i}_{diff[0]}_{diff[1]}_{'_'.join(classes)}.npz", dataset)













