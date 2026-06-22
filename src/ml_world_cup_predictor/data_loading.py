import shutil
import kagglehub
import pandas as pd
from pathlib import Path


def get_data_path(data_directory:Path) -> Path:

    if data_directory.exists():
        path = data_directory
        print("Using cached dataset at:", path)
    else:
        download_path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
        data_directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(download_path, data_directory)
        path = data_directory
        print("Downloaded dataset to:", path)

    return path


def load_raw_data(data_directory: Path) -> dict[str,pd.DataFrame]:
    frames = {file_path.stem : pd.read_csv(file_path) for file_path in sorted(data_directory.glob('*csv'))}

    for frame in frames.values():
        frame = _assign_date(frame)    
    return frames

def load_processed_data(processed_directory:Path) -> pd.DataFrame:

    df = pd.read_csv(processed_directory)    

    return _assign_date(df)


def _assign_date(df:pd.DataFrame) -> pd.DataFrame:

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
