import numpy as np 
import pandas as pd


class DataLoader:
    def __init__(self, data_config) -> None:
        self._data_config = data_config
    
    def load_data(self):
        df = pd.read_csv(self._data_config["path"])
        return df


class SequenceSplitter:
    def split_sequences(self, df, seq_length):
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(df[i:i+seq_length])
            y.append(df[i+seq_length])
            
        X = np.array(X)
        y = np.array(y)
        return X, y


class DataPreprocessor:
    def __init__(self, data_config) -> None:
        self._data_config = data_config
    
    def preprocess_data(self, dataset, seq_length):
        train_size = int(self._data_config["train_size"] * len(dataset))
        validation_size = int(self._data_config["validation_size"] * len(dataset))
        # test_size = int(self._data_config["test_size"]*len(dataset))

        train_data = dataset[:train_size]
        val_data = dataset[train_size:validation_size]
        test_data = dataset[train_size+validation_size:]

        return train_data, val_data, test_data


class NNDataLoader:
    def __init__(self, data_config) -> None:
        self._data_config = data_config
        self._data_loader = DataLoader(data_config)
        self._sequence_splitter = SequenceSplitter()
        self._data_preprocessor = DataPreprocessor(data_config)
    
    def load_sequences(self, seq_length):
        df = self._data_loader.load_data()
        train_data, val_data, test_data = self._data_preprocessor.preprocess_data(df, seq_length)
        X_train, y_train = self._sequence_splitter.split_sequences(train_data, seq_length)
        X_val, y_val = self._sequence_splitter.split_sequences(val_data, seq_length)
        X_test, y_test = self._sequence_splitter.split_sequences(test_data, seq_length)
        return X_train, X_val, X_test, y_train, y_val, y_test