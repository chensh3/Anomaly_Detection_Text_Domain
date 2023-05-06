# import basic package
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
dataset_list = [x.split(".npz")[0] for x in os.listdir('D:/anomaly_data')]
dataset_list = sorted(dataset_list)

model_dict = { 'COPOD': PYOD, 'DeepSVDD': PYOD}  # WORKS


# save the results
df_AUCROC = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())
df_AUCPR = pd.DataFrame(data = None, index = dataset_list, columns = model_dict.keys())

# seed for reproducible results
seed = 42

for i, dataset in tqdm(enumerate(dataset_list)):
    '''
    la: ratio of labeled anomalies, from 0.0 to 1.0
    realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
    noise_type: inject data noises for testing model robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
    '''
    if pd.isna(pd.read_pickle("results_aucpr.pickle").loc[dataset].COPOD):
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
            clf = clf.fit(X_train = data['X_train'],y_train= data['y_train'])

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
    else:
        print("already done with:",dataset)
print(df_AUCROC)
print(df_AUCPR)
df_AUCPR.to_pickle("results_aucpr.pickle")
df_AUCROC.to_pickle("results_aucroc.pickle")
print(f"Avg score:\n\tAUCROC:\n{df_AUCROC.mean().to_string()}\n\tAUCPR:\n{df_AUCPR.mean().to_string()}")
# firzt Semi supervised
# print(df_AUCROC)
# print(df_AUCPR)
# 0.3 test size
#               ECOD     COPOD
# 20news_0  0.621832  0.629881
# 20news_1  0.474321  0.479643
# 20news_2  0.472414  0.446601
# 20news_3  0.626136  0.640341
# 20news_4  0.523938   0.49668
# 20news_5  0.542093  0.541371
# agnews_0  0.497594  0.513114
# agnews_1  0.563161  0.526156
# agnews_2  0.608542  0.615284
# agnews_3  0.539375  0.542109
# amazon    0.536911  0.567149
# imdb      0.462683  0.504766
# yelp      0.567064  0.593046
#               ECOD     COPOD
# 20news_0  0.193422   0.19876
# 20news_1  0.139538  0.139814
# 20news_2  0.139988  0.131979
# 20news_3  0.218563  0.251924
# 20news_4  0.190839  0.171422
# 20news_5  0.182496  0.172804
# agnews_0  0.147114  0.154638
# agnews_1   0.17408  0.148465
# agnews_2  0.198809  0.214843
# agnews_3  0.157881  0.163024
# amazon    0.160264  0.171431
# imdb      0.131537  0.143646
# yelp      0.180873  0.195891

# 2n test size ( n- number of anomalies)
#  AUCROC     ECOD     COPOD
# 20news_0  0.580094  0.578196
# 20news_1  0.508096  0.497088
# 20news_2  0.482765   0.45571
# 20news_3  0.627778  0.601111
# 20news_4   0.49591  0.477171
# 20news_5  0.488227  0.481821
# agnews_0   0.49718   0.50916
# agnews_1  0.566776  0.523924
# agnews_2  0.610552  0.600788
# agnews_3  0.552412  0.555148
# amazon     0.53378   0.56646
# imdb        0.4625  0.500916
# yelp      0.555772  0.580416
#  AUCPR      ECOD     COPOD
# 20news_0  0.542667  0.537817
# 20news_1  0.514017  0.501251
# 20news_2  0.513738   0.48548
# 20news_3  0.635696  0.616853
# 20news_4  0.559258  0.535439
# 20news_5  0.514473  0.491853
# agnews_0  0.499014  0.509145
# agnews_1  0.541234  0.494018
# agnews_2  0.591775  0.603231
# agnews_3  0.526888  0.536154
# amazon    0.516538  0.535657
# imdb      0.462225  0.489265
# yelp      0.546601  0.566341
