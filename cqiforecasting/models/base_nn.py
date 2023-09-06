"""Abstract base model"""

from abc import ABC, abstractmethod
from cqiforecasting.utils.config import Config
from cqiforecasting.dataloader.nn_data_loader  import NNDataLoader

class BaseNN(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, cfg):
        self._model = None 
        self.eval = None 
        self.config = Config.from_json(cfg)
        self.data_loader = NNDataLoader(self.config["data"])

    def load_data(self):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.data_loader.load_sequences(self._seq_length)

    @abstractmethod
    def build(self):
        pass
    
    @abstractmethod
    def compile(self):
        pass

    def train(self, batch_size, epochs):
        self._model.compile()
        history = self._model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=(self.X_val, self.y_val))

        return history
    
    def evaluate(self):
        self.eval = self._model.evaluate(self.X_test, self.y_test)

        return self.eval        

    