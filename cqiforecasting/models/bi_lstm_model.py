from .base_nn import BaseNN
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM
from keras.losses import MeanSquaredError
from keras.optimizers import Adam 


class BiLSTMModel(BaseNN):

    def __init__(self, input_seq_length, cfg, n_features=1, name="BiLSTM"):
        super().__init__(input_seq_length, cfg, n_features)
        
        self.name = name

        self._description = "Double layer LSTM neural net"

   

        self._LSTM1_units = 25
        self._LSTM2_units = 25


    def build(self):
        self._model = Sequential()
        #adding layers
        self._model.add(Bidirectional(LSTM(self._LSTM1_units, input_shape=(self._seq_length, self._n_features),activation="relu",return_sequences=True)))
        self._model.add(Bidirectional(LSTM(self._LSTM2_units, activation="relu")))
        self._model.add(Dense(1))
    
    def compile(self):
        loss = MeanSquaredError()
        optimizer = Adam()
        self._model.compile(loss=loss, optimizer=optimizer)

