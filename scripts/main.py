from cqiforecasting.configs.config import CFGLog
from cqiforecasting.dataloader.nn_data_loader import NNDataLoader



if __name__ == "__main__":
    df = NNDataLoader().load_data(CFGLog["data"])
    print(df.head())
