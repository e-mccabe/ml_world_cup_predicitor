import hashlib
import json
import shutil
import kagglehub
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from ml_world_cup_predictor.config import DATA_DIRECTORY, PROCESSED_DIRECTORY,KAGGLE_LINK,PATH



def process_raw_data(path,refresh:bool = False,**schema) -> dict[str,pd.DataFrame]:
    """Takes any files in the raw folder, handles them via a passed schema and writes to interim directory"""

    # Ensure the files exist in the /raw directory or load in new files
    _check_raw_directory(path.raw,refresh)

    # Issues with the **schema input, the inpurs will have to be per file to work

    raw_files = _load_folder_to_dict(path.raw,**schema)

    # Writing dictionary of files to interim folder
    _write_to_interim(raw_files,path.interim)

    print(f'Files load {raw_files.keys()}')
    return raw_files


def _check_raw_directory(path:Path,refresh:bool = False):
    """Looks in the raw directory for files, if it is empty load from kaggle and add to the raw directory"""
    path.mkdir(exist_ok=True,parents=True)

    if any(path.iterdir()) and not refresh:
        print(f'Raw files available to be loaded: {path}')
    
    else:
        download_path = kagglehub.dataset_download(KAGGLE_LINK)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(download_path, path,dirs_exist_ok=True)
        print("Downloaded dataset to raw directory:", path)



def _load_folder_to_dict(path:Path,**kwargs):
    """Loads all csv files in the provide path into a dictionary, with the file names as key"""
    df_dict =  {file_path.stem : pd.read_csv(file_path, **kwargs) for file_path in sorted(path.glob('*csv'))}
    if not df_dict:
        raise FileNotFoundError(f'No Files Found in: {path}')
    
    return df_dict

def _write_to_interim(df_dict:dict[str,pd.DataFrame],path:Path) -> None:
    """Takes the dictionary of dataframes with the correct schema and writes to interim folder"""

    path.mkdir(exist_ok=True,parents=True)

    for key,value in df_dict.items():
        try:
            write_path = path / f'{key}.parquet'
            value.to_parquet(write_path)
            print(f'Files written to {path}')
        except Exception as e:
            write_path = path / f'{key}.csv'
            value.to_csv(write_path)
            print(f'Files written to {path}')


######################################## OLD FUNCTIONS (Still in Use) ######################################## 

def get_data_path(data_directory:Path,refresh:bool = False) -> Path:
    """Checks if raw files exist in the directory or if a download from kaggle is required. Returns file paths to raw files"""

    # Check if the directory is empty
    is_empty = not any(data_directory.iterdir())

    # If the directory is not empty the load from the data available locally
    if not is_empty and data_directory.exists() and not refresh:
        path = data_directory
        print("Using cached dataset at:", path)
    
    # Otherwise download from kaggle and create the dataset locally
    else:
        download_path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
        data_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(download_path, data_directory,dirs_exist_ok=True)
        path = data_directory
        print("Downloaded dataset to:", path)

    return path

def load_raw_data(data_directory: Path,refresh:bool = False) -> dict[str,pd.DataFrame]:
    """"""
    # Identify the file path for the dataframes
    correct_path = get_data_path(data_directory=data_directory,refresh=refresh)

    # Load the frames into a dictionary
    frames = {file_path.stem : pd.read_csv(file_path) for file_path in sorted(correct_path.glob('*csv'))}
    
    
    if not frames:
        raise ValueError(f'No files found at {correct_path}')

    # Clean up any potrntial date columns
    for frame in frames.values():
        frame = _assign_date(frame)    
    return frames


def load_processed_data(processed_directory:Path) -> pd.DataFrame:

    df = pd.read_csv(processed_directory)    

    return _assign_date(df)


def _assign_date(df:pd.DataFrame) -> pd.DataFrame:

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')


def load_data(directory:Path = PROCESSED_DIRECTORY, refresh:bool = False) -> dict[str:pd.DataFrame]:

    if any(directory.iterdir()) and not refresh:
        return {file_path.stem : load_processed_data(file_path) for file_path in sorted(directory.glob('*csv'))}
    else:
        frames = load_raw_data(DATA_DIRECTORY,refresh)
        return frames


if __name__ == '__main__':

    process_raw_data(PATH)