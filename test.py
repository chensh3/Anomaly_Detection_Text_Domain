# import basic package
import os
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

# import the necessary package
from data_generator import DataGenerator
from myutils import Utils

datagenerator = DataGenerator()  # data generator
utils = Utils()  # utils function

# from baseline.PyOD import PYOD
from ADBench.baseline.PyOD import PYOD
from baseline.GANomaly.run import GANomaly
from baseline.FEAWAD.run import FEAWAD
from baseline.REPEN.run import REPEN
# dataset and model list / dict
dataset_list = [x.split(".npz")[0] for x in os.listdir('datasets/NLP_by_BERT')]
dataset_list = sorted(dataset_list)
# dataset_list=['20news_5']

# model_dict = {'FEAWAD':FEAWAD,'GANomaly': GANomaly}
model_dict = {'XGBOD': PYOD, 'GANomaly': GANomaly}
# save the results
df_AUCROC = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())
df_AUCPR = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())

# seed for reproducible results
seed = 42

for i, dataset in enumerate(dataset_list):
    '''
    la: ratio of labeled anomalies, from 0.0 to 1.0
    realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
    noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
    '''
    print("\n\n")
    print(f"#{i} : Dataset Name: {dataset}")
    # import the dataset
    datagenerator.dataset = dataset  # specify the dataset name
    data = datagenerator.generator(la = -1, realistic_synthetic_mode = None,
                                   noise_type = None)  # only 0% labeled anomalies are available

    for name, clf in model_dict.items():
        print("model:", name)
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

print(df_AUCROC)
print(df_AUCPR)

# firzt Semi supervised
# print(df_AUCROC)
# print(df_AUCPR)
#           GANomaly
# 20news_0  0.599113
# 20news_1   0.54357
# 20news_2  0.506082
# 20news_3   0.60928
# 20news_4  0.544169
# 20news_5  0.556516
# agnews_0  0.564224
# agnews_1  0.628938
# agnews_2   0.62025
# agnews_3  0.627627
# amazon    0.588033
# imdb       0.50702
# yelp      0.660044
#           GANomaly
# 20news_0  0.191747
# 20news_1   0.16721
# 20news_2  0.147619
# 20news_3  0.243205
# 20news_4   0.17425
# 20news_5  0.191426
# agnews_0  0.173924
# agnews_1  0.203089
# agnews_2   0.20869
# agnews_3  0.209381
# amazon    0.182335
# imdb      0.146117
# yelp      0.221336
