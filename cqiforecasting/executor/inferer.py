import numpy as np 

class Inferer:
    @staticmethod
    def infer(model, input_sequences) -> np.array:
        output = model.predict(input_sequences)
        return output
    
    @staticmethod
    def infer_steps(model, input_sequence, num_steps=1) -> np.array:
        x_forecast = np.array([input_sequence])
        y_forecast = np.array([[0],])

        for _ in range(num_steps):
            p = model.predict(np.array([x_forecast[-1],]))
            y_forecast = np.append(y_forecast, p, axis=0)
            new_arr = x_forecast[-1].copy()
            new_arr[:-1] = new_arr[1:]
            new_arr[-1, :] = y_forecast[-1]
            x_forecast = np.append(x_forecast, [new_arr,], axis=0)

        y_forecast = y_forecast[1:]

        return y_forecast