from .base_nn import BaseNN
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.losses import Huber 
from keras.optimizers import Adam 

class LSTMModel(BaseNN):

    def __init__(self, input_seq_length, cfg, n_features=1, name="LSTM"):
        super().__init__(cfg)
        
        self.name = name

        self._description = "Double layer LSTM neural net"

        self._seq_length = input_seq_length
        self._n_features = n_features
        
        self._LSTM1_units = 50
        self._LSTM2_units = 50


    def build(self):
        self._model = Sequential()
        #adding layers
        self._model.add(LSTM(self._LSTM1_units, input_shape=(self._seq_length, self._n_features),activation="relu",return_sequences=True))
        self._model.add(LSTM(self._LSTM2_units, activation="relu"))
        self._model.add(Dense(1))
    
    def compile(self):
        loss = Huber()
        optimizer = Adam()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])

   

   


      
