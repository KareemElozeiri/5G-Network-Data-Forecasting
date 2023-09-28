"""Model configurations in json format. 
Paths are always relative to the location of the command line,
paths are not relative to any objects in the source codes.
If you are not running scripts from the root folder (e.g., using cd instead), 
then the paths here need to be checked and modified accordingly"""
import os

# "./" = root folder of the project
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data/')
export_dir  = os.path.join(current_dir, '../output/exported_models')
CFGLog = {
    "data": {
        "path": os.path.join(data_dir,"UE1-CQI/UE1-CQI_1.csv"),
        "train_size":0.7,
        "validation_size":0.15,
        "test_size":0.15,
    },
    "train": {
            
    },
    "output": {
        "models_output_path": os.path.join(export_dir, "/models/"),
        "reports_output_path": os.path.join(export_dir, "/reports/")
    }
}