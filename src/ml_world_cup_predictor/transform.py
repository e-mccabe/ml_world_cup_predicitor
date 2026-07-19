import pandas as pd
import numpy as np

from ml_world_cup_predictor.config import Paths
from ml_world_cup_predictor.utils import load_folder_to_dict, write_to_folder
from ml_world_cup_predictor.load import process_raw_data

def transform_datasets(path:Paths,refresh:bool = False) -> dict[str,pd.DataFrame]:
    """Orchestrates the generation of processed dataframes through mapping, conditional assignment and splitting by played & not played"""

    # If no refresh passed and the interim directory is not empty, load the files
    if not refresh and any(path.interim.iterdir()):
        print(f'Loading from {path.interim}')
        dataframe_dict = load_folder_to_dict(path.interim)

    # Catches refresh, loads from raw and writes to interim
    else:
        dataframe_dict = process_raw_data(path,refresh)

    processed = enrich_datasets(dataframe_dict)

    # Writing processed files to the processed directory
    write_to_folder(processed,path.processed)

    return processed    
    
def enrich_datasets(dataframe_dict:dict[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """Joins datasets and adds columns required for downstream use"""
    joined_df = _join_datasets(dataframe_dict)
    joined_df['year'] = joined_df['date'].dt.year
    joined_df['goal_diff'] = np.abs(joined_df['home_score'] - joined_df['away_score'])

    results_built_df = _create_results_column(joined_df)


    not_played = results_built_df[results_built_df['home_score'].isna()].copy()
    played = results_built_df[~results_built_df['home_score'].isna()].copy()

    return {'played':played, 'not_played':not_played} 

def _join_datasets(dataframe_dict: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Maps the shootout and former names data to results"""

    results = dataframe_dict['results'].copy()
    
    ### ----- Mapping Shootout -----
    shootouts = dataframe_dict['shootouts'].copy()

    # Generate an id shared across both datasets
    results['shootout_id'] = (results['date'].astype(str) + '_' + results['home_team'] + '_' + results['away_team'])
    shootouts['shootout_id'] = (shootouts['date'].astype(str) + '_' + shootouts['home_team'] + '_' + shootouts['away_team'])

    # Convert shootout winner to mapping dict
    shootout_dict = pd.Series(shootouts['winner'].values, index = shootouts['shootout_id']).to_dict()

    results['shootout_winner'] = results['shootout_id'].map(shootout_dict)
    results = results.drop('shootout_id',axis = 1)
    
    ### ----- Updating Former Names -----
    former_names = dataframe_dict['former_names']
    name_map = dict(zip(former_names['former'],former_names['current']))

    results['home_team'] = results['home_team'].replace(name_map)
    results['away_team'] = results['away_team'].replace(name_map)

    return results

def _create_results_column(df:pd.DataFrame) -> pd.DataFrame:
    """Creates column based of scores & shootout winner columns"""
    df = df.copy()

    match_conditions = [
    df['home_score'] > df['away_score'],
    df['home_score'] < df['away_score'],
    df['shootout_winner'] == df['home_team'],
    df['shootout_winner'] == df['away_team'],
    (df['home_score'] == df['away_score']) & df['shootout_winner'].isna()
    ]  

    choices = ['W','L','W','L','D']
    df['result'] = np.select(match_conditions,choices,default=None)

    return df
