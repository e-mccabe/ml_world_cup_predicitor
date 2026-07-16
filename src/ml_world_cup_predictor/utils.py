from pathlib import Path
import pandas as pd

from ml_world_cup_predictor.config import MINIMUM_GAMES_PLAYED


def load_folder_to_dict(path:Path,**kwargs):
    """Loads all csv files in the provide path into a dictionary, with the file names as key"""
    df_dict =  {file_path.stem : pd.read_csv(file_path,parse_dates=True, **kwargs) for file_path in sorted(path.glob('*csv'))}

    if not df_dict:
        raise FileNotFoundError(f'No Files Found in: {path}')

    for label, df in df_dict.items():
        df_dict[label] = assign_date_columns(df)

    return df_dict

def write_to_folder(df_dict:dict[str,pd.DataFrame],path:Path) -> None:
    """Takes the dictionary of dataframes and writes to a specified folder"""
    # Make sure path exists first before writing the files
    path.mkdir(exist_ok=True,parents=True)

    # Go through writing each file, try parquet first, fall back to csv
    for key,value in df_dict.items():
        try:
            write_path = path / f'{key}.parquet'
            value.to_parquet(write_path)
            print(f'Files written to {path}')
        except Exception as e:
            write_path = path / f'{key}.csv'
            value.to_csv(write_path)
            print(f'Files written to {path}')


def generate_team_list(df:pd.DataFrame,minimum_games:int = MINIMUM_GAMES_PLAYED) -> list[str]:
    """Helper function to generate a unique list of Team names from the current dataset"""
    df = df.copy()
    
    df = df.melt(
    id_vars=['date','year','tournament'],
    value_vars=['home_team','away_team'],
    value_name='team'
)
    count = df['team'].value_counts()
    count = count[count >= minimum_games]

    return sorted(list(count.index))

def assign_date_columns(df:pd.DataFrame) -> pd.DataFrame:

    date_cols = [col for col in df.columns if 'date' in col]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df
