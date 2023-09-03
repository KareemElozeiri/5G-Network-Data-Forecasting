"""Model configurations in json format. 
Paths are always relative to the location of the command line,
paths are not relative to any objects in the source codes.
If you are not running scripts from the root folder (e.g., using cd instead), 
then the paths here need to be checked and modified accordingly"""


# "./" = root folder of the project
CFGLog = {
    "data": {
        "path": "./data/ue-lte-network-traffic-stats.csv",

    },
    "train": {
            
    },
    "output": {
        "output_path": "./exported_models/",
    }
}