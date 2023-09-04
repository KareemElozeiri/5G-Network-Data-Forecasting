import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

'''script start here'''
from cqiforecasting.configs.config import CFGLog
from cqiforecasting.dataloader.nn_data_loader import NNDataLoader




if __name__ == "__main__":
    df = NNDataLoader().load_data(CFGLog["data"])
    print(df.head())