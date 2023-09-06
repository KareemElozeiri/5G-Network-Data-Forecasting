import os
import sys


class PostProcessor:

    @staticmethod
    def save_model_results(model, output_config):
        file_path = os.path.join(output_config["reports_output_path"], f"{model.name}_info")

        loss = model.eval["loss"]
        mae = model.eval["mae"]

        with open(file_path, 'w') as f:

            f.write(f"Loss: {loss}\n")
            f.write(f"MAE: {mae}\n")

            sys.stdout = f
            model.summary()
            sys.stdout = sys.__stdout__
    
    @staticmethod
    def save_model(model, output_config):
        file_path = os.path.join(output_config["reports_output_path"], f"{model.name}_weights")
        model.model.save(file_path)