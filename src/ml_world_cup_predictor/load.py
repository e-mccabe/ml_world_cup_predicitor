import shutil
import kagglehub
import pandas as pd
from pathlib import Path

from ml_world_cup_predictor.config import KAGGLE_LINK, Paths
from ml_world_cup_predictor.utils import load_folder_to_dict, write_to_folder

def process_raw_data(path:Paths,refresh:bool = False,**schema) -> dict[str,pd.DataFrame]:
    """Takes any files in the raw folder, handles them via a passed schema and writes to interim directory"""
    # Ensure the files exist in the /raw directory or load in new files
    _check_raw_directory(path.raw,refresh)

    # TODO: Issues with the **schema input, the inputs will have to be per file to work

    raw_files = load_folder_to_dict(path.raw,**schema)

    # Writing dictionary of files to interim folder
    write_to_folder(raw_files,path.interim)

    return raw_files

def _check_raw_directory(path:Path,refresh:bool = False):
    """Looks in the raw directory for files, if it is empty load from kaggle and add to the raw directory"""
    
    # Create directory if it doesn't already exist
    path.mkdir(exist_ok=True,parents=True)

    # If the folder is not empty and no force refresh is passed
    if any(path.iterdir()) and not refresh:
        print(f'Raw files available to be loaded: {path}')
    
    # If the folder is empty or a force refresh is passed
    else:
        # Link to kaggle dataset
        download_path = kagglehub.dataset_download(KAGGLE_LINK)
        # Copies downloaded files from initial location to project folder
        shutil.copytree(download_path, path,dirs_exist_ok=True)
        print("Downloaded dataset to raw directory:", path)
