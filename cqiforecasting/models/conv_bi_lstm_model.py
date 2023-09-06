from .base_nn import BaseNN
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Conv1D
from keras.losses import Huber 
from keras.optimizers import Adam 


class ConvBiLSTMModel(BaseNN):

    def __init__(self, input_seq_length, cfg, n_features=1):
        super().__init__(cfg)
        self._description = "Double layer LSTM neural net"
        
        self._seq_length = input_seq_length
        self._n_features = n_features
        
        self._n_filters = 64
        self._kernel_size = 3
        self._strides = 1
        self._padding = 'causal'
        
        self._LSTM1_units = 32
        self._LSTM2_units = 32


    def build(self):
        self._model = Sequential()
        #adding layers
        self._model.add(Conv1D(filters=self._n_filters, kernel_size=self._kernel_size, strides=self._strides, activation="relu", padding=self._padding, input_shape=[self._seq_length, self._n_features]))
        self._model.add(Bidirectional(LSTM(self._LSTM1_units, input_shape=(self._seq_length, self._n_features),activation="relu",return_sequences=True)))
        self._model.add(Bidirectional(LSTM(self._LSTM2_units, activation="relu")))
        self._model.add(Dense(30, activation="relu"))
        self._model.add(Dense(10, activation="relu"))
        self._model.add(Dense(1))
    
    
    def compile(self):
        loss = Huber()
        optimizer = Adam()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

  
