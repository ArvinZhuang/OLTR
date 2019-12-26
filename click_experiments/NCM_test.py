from clickModel.NCM import NCM
from utils import read_file as rf
import numpy as np

model = NCM(1, 100, 10240+1024+1, 2)

click_log_path = "../feature_click_datasets/{}/train_set{}.txt".format("SDBN", 1)

click_log = rf.read_click_log(click_log_path)

model.initial_representation(click_log)

# session = np.array(['1112', '16', '3', '45', '37', '31', '22', '5', '34', '17', '21', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0' ])

model.train(click_log)