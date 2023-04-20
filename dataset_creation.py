import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
import itertools

data = pd.read_json("News_Category_Dataset_v3.json", lines = True)
data = data[["headline", "category"]]
data.category = data.category.str.lower()

data.loc[(data.category == "the worldpost") | (data.category == "worldpost"), 'category'] = "world news"
data.loc[(data.category == "style & beauty"), 'category'] = "style"
data.loc[(data.category == "arts") | (data.category == "culture & arts"), 'category'] = "arts & culture"
data.loc[(data.category == "parents"), 'category'] = "parenting"
data.loc[(data.category == "taste"), 'category'] = "food & drink"
data.loc[(data.category == "green"), 'category'] = "environment"
data.loc[(data.category == "healthy living"), 'category'] = "healthy living tips"

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


easy = results[0:2]
medium = results[3:5]
hard = results[7:]
easy_df=pd.DataFrame(columns=data.columns)
medium_df=pd.DataFrame(columns=data.columns)
hard_df=pd.DataFrame(columns=data.columns)
df=[easy_df,medium_df,hard_df]

# for i,type_data in enumerate([easy,medium,hard]):
#     for diff in type_data:
diff=easy[0]
if diff[0][0]!=diff[0][1]:
    classes=vectors.loc[(vectors.bin==diff[0][0]) | (vectors.bin==diff[0][1]),"classes" ].values
    classes=np.hstack(classes)[0]
    anomaly = data.loc[(data.category==classes[0]) | (data.category==classes[1]),:]
    anomaly.loc[:,"label"] =np.ones(len(anomaly))
    normal = data.loc[(data.category!=classes[0]) | (data.category!=classes[1]),:]
    normal.loc[:,"label"]  = np.zeros(len(normal))
    noraml_to_test=normal.sample(len(anomaly))
    # anomaly["label"] = anomaly.category.apply(lambda x: 1 if x == classes[0] or x == classes[1] else 0)

    X_test = anomaly.append(noraml_to_test)
    X_train = normal.drop(noraml_to_test.index)
    y_train = X_train.loc[:,["label"]]
    y_test = X_test.loc[:,["label"]]
    X_test = X_test.loc[:,["headline","category"]]
    X_train = X_train.loc[:,["headline","category"]]

    dataset={'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
    np.save(f"{diff}_{classes}.npz", dataset)




















