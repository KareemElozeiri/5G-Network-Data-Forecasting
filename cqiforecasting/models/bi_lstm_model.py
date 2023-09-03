from .base_model import BaseModel


class BiLSTMModel(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._description = "Double layer Bi-direcational LSTM neural net"

    def load_data(self):
        return super().load_data()

    def build(self):
        return super().build()

    def train(self):
        return super().train()

    def evaluate(self):
        return super().evaluate()

    def evaluate_document(self):
        return super().evaluate_document()    