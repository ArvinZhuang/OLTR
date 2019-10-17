from dataset.LetorDataset import LetorDataset

dataset = "../datasets/test.txt"

t = LetorDataset(dataset, 46)
print(len(t.get_all_querys()))