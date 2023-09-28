import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self, data_config) -> None:
        self._data_config = data_config
    
    def load_data(self):
        df = pd.read_csv(self._data_config.path)
        return df["UE1-CQI"]


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
        train_size = int(self._data_config.train_size * len(dataset))
        validation_size = int(self._data_config.validation_size * len(dataset))
        # test_size = int(self._data_config["test_size"]*len(dataset))

        train_data = dataset[:train_size].values
        val_data = dataset[train_size:train_size+validation_size].values
        test_data = dataset[train_size+validation_size:].values

        return train_data, val_data, test_data


class NNDataLoader:
    def __init__(self, data_config) -> None:
        self._data_config = data_config
        self._data_loader = DataLoader(data_config)
        self._sequence_splitter = SequenceSplitter()
        self._data_preprocessor = DataPreprocessor(data_config)
        self._scaler = MinMaxScaler()
    
    def reshape_feature_vector(self, n_features, feat_vect):
        return feat_vect.reshape((feat_vect.shape[0], feat_vect.shape[1], n_features))

    def load_sequences(self, seq_length, n_features, scale=True):
        df = self._data_loader.load_data()
        train_data, val_data, test_data = self._data_preprocessor.preprocess_data(df, seq_length)
        if scale:
            train_data, val_data, test_data = self._scaler.fit_transform(train_data.reshape(-1, 1)), self._scaler.fit_transform(val_data.reshape(-1, 1)), self._scaler.fit_transform(test_data.reshape(-1, 1))
        X_train, y_train = self._sequence_splitter.split_sequences(train_data, seq_length)
        X_train = self.reshape_feature_vector(n_features, X_train)        

        X_val, y_val = self._sequence_splitter.split_sequences(val_data, seq_length)
        X_val = self.reshape_feature_vector(n_features, X_val)        

        X_test, y_test = self._sequence_splitter.split_sequences(test_data, seq_length)
        X_test = self.reshape_feature_vector(n_features, X_test)        

        
        return X_train, X_val, X_test, y_train, y_val, y_test