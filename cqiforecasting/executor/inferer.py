import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class Inferer:
    @staticmethod
    def infer(model, input_sequences) -> np.array:
        scaler = MinMaxScaler()
        sequence_shape = input_sequences.shape
        scaled_input = scaler.fit_transform(input_sequences.reshape(-1, 1))
        scaled_input = scaled_input.reshape(sequence_shape)
        print(scaled_input.shape)
        output = model.predict(scaled_input)
        unscaled_output = scaler.inverse_transform(output).squeeze()
        return unscaled_output
    
    @staticmethod
    def infer_steps(model, input_sequence, num_steps=1) -> np.array:
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(input_sequence.reshape(-1, 1))
        x_forecast = np.array([scaled_input])
        y_forecast = np.array([[0],])

        for _ in range(num_steps):
            p = model.predict(np.array([x_forecast[-1],]))
            y_forecast = np.append(y_forecast, p, axis=0)
            new_arr = x_forecast[-1].copy()
            new_arr[:-1] = new_arr[1:]
            new_arr[-1, :] = y_forecast[-1]
            x_forecast = np.append(x_forecast, [new_arr,], axis=0)

        y_forecast = y_forecast[1:]
        unscaled_y_forecast = scaler.inverse_transform(y_forecast)
        return unscaled_y_forecast