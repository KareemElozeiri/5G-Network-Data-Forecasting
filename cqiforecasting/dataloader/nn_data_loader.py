import numpy as np 
import pandas as pd


class NNDataLoader:
    
    def __init__(self, data_config) -> None:
        self._data_config = data_config
    
    def load_data(self):
        df = pd.read_csv(self._data_config["path"])
        return df

    def split_sequences(self, df, seq_length):
        X, y = [], []
        for i in range(len(df)-seq_length):
            X.append(df[i:i+seq_length])
            y.append(df[i+seq_length])
            
        X = np.array(X)
        y = np.array(y)
        return X, y

    def preprocess_data(self, dataset, seq_length):

        train_size = int(self._data_config["train_size"]*len(dataset))
        validation_size = int(self._data_config["validation_size"]*len(dataset))  
        #test_size = int(self._data_config["test_size"]*len(dataset))

        train_data = dataset[:train_size]
        val_data = dataset[train_size:validation_size]
        test_data = dataset[train_size+validation_size:]

        X_train, y_train = self.split_sequences(train_data, seq_length)
        X_val, y_val = self.split_sequences(val_data, seq_length)
        X_test, y_test = self.split_sequences(test_data, seq_length)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_sequences(self, seq_length):
        df = self.load_data()

        return self.preprocess_data(df, seq_length)

