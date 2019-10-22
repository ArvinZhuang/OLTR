from dataset import LetorDataset
from ranker import  PDGDLinearRanker
num_features = 46
Learning_rate = 0.1
dataset = LetorDataset("../datasets/test.txt", num_features)
#%%

ranker = PDGDLinearRanker(num_features, Learning_rate)
