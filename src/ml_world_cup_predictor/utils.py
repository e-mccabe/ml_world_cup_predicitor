from pathlib import Path
import pandas as pd
import numpy as np

from ml_world_cup_predictor.config import MINIMUM_GAMES_PLAYED

def load_folder_to_dict(path:Path,**kwargs):
    """Loads all csv files in the provided path into a dictionary, with the file names as key"""

    df_dict =  {file_path.stem : pd.read_csv(file_path, **kwargs) for file_path in sorted(path.glob('*csv'))}

    if not df_dict:
        raise FileNotFoundError(f'No Files Found in: {path}')

    for df_name, df in df_dict.items():
        df_dict[df_name] = assign_date_columns(df)

    return df_dict

def write_to_folder(df_dict:dict[str,pd.DataFrame],path:Path) -> None:
    """Takes the dictionary of dataframes and writes to a specified folder"""
    # Make sure path exists first before writing the files
    path.mkdir(exist_ok=True,parents=True)

    # Go through writing each file, try parquet first, fall back to csv
    for df_name,df in df_dict.items():
        try:
            write_path = path / f'{df_name}.parquet'
            df.to_parquet(write_path)
            print(f'Files {df_name} written to parquet {path}')
        except Exception as error:
            print(f'Parquet file write failed: {error}')
            write_path = path / f'{df_name}.csv'
            df.to_csv(write_path)
            print(f'File {df_name} written to csv {path}')

# TODO: replace with a new minimum-games filter function that does not use date, year or tournament

def generate_team_list(df:pd.DataFrame,minimum_games:int = MINIMUM_GAMES_PLAYED) -> list[str]:
    """Generate a unique list of Team names from the current dataset. """
    df = df.copy()
    
    df = df.melt(
    id_vars=['date','year','tournament'],
    value_vars=['home_team','away_team'],
    value_name='team'
)
    team_count = df['team'].value_counts()
    team_count = team_count[team_count >= minimum_games]

    return sorted(list(team_count.index))

def assign_date_columns(df:pd.DataFrame) -> pd.DataFrame:
    """Identifies any columns with 'date' in the name and convert to datetime value"""
    date_cols = [column for column in df.columns if 'date' in column]

    for column in date_cols:
        df[column] = pd.to_datetime(df[column], errors='coerce')

    return df

def chronological_split(df:pd.DataFrame,split_ratio:float = 0.8) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Splits the feature dataframe into a training set and a test set. Assume df is sort chronologically"""
    
    dataframe_length = len(df)

    cut_off_index = int(dataframe_length*split_ratio)

    train = df.iloc[:cut_off_index]
    test = df.iloc[cut_off_index:]

    return train,test


def get_long_team_df(df:pd.DataFrame)-> pd.DataFrame:

    df = df.reset_index(names = 'match_id').copy()

    base_columns = ['match_id','date','home_team','away_team','home_score','away_score','result']

    home_map = {
        'home_team':'team',
        'away_team':'opponent',
        'home_score':'goals_scored',
        'away_score':'goals_conceded'
    }
    
    away_map = {
        'home_team':'team',
        'away_team':'opponent',
        'home_score':'goals_scored',
        'away_score':'goals_conceded'
    }

    home = _get_side_dataframe(df,base_columns,home_map,'W')
    away = _get_side_dataframe(df,base_columns,away_map,'L')

    return pd.concat([home,away]).sort_values(['date','match_id'],kind = 'stable')

def _get_side_dataframe(df:pd.DataFrame,base_colums:list[str],column_map:dict[str,str],win_sign:str)->pd.DataFrame:

    df = df.copy()

    df = df[base_colums].rename(columns = column_map)
    df['win'] = (df['result'] == win_sign).astype(float)
    df['side'] = 'home' if win_sign == 'W' else 'away'

    return df