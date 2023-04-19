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























