import pandas as pd
import numpy as np
from pathlib import Path


from ml_world_cup_predictor.config import DATA_DIRECTORY,PROCESSED_DIRECTORY,PATH
from ml_world_cup_predictor.utils import load_folder_to_dict, write_to_folder
from ml_world_cup_predictor.load import load_raw_data, process_raw_data


def transform_datasets(path,refresh:bool = False) -> dict[str,pd.DataFrame]:
    """Orchestrates the generation of processed dataframes through mapping, conditional assignment and splitting by played & not played"""

    # If no refresh passed and the interim directory is not empty, load the files    
    if not refresh and any(path.interim.iterdir()):
        print(f'Loading from {path.interim}')
        dataframe_dict = load_folder_to_dict(path.interim)

    else:
        dataframe_dict = process_raw_data(path,refresh)    

    processed = enrich_datasets(dataframe_dict)

    # Writing processed files to the processed directory
    write_to_folder(processed,path.processed)

    return processed    
    
def enrich_datasets(dataframe_dict:dict[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """Join datasets and adds columns required for downstream use"""
    
    
    joined_df = _join_datasets(dataframe_dict)
    joined_df['year'] = joined_df['date'].dt.year
    joined_df['goal_diff'] = joined_df['home_score'] - joined_df['away_score']
    draw_df = _create_results_column(joined_df)


    not_played = draw_df[draw_df['home_score'].isna()].copy()
    played = draw_df[~draw_df['home_score'].isna()].copy()

    return {'played':played, 'not_played':not_played} 

def _join_datasets(dataframe_dict: dict[str,pd.DataFrame]) -> pd.DataFrame:
    """Maps the shooutout and former names data to results"""
    
    results = dataframe_dict['results']
    
    ### ----- Mapping Shootout -----
    shootouts = dataframe_dict['shootouts']

    # Generate an id shared across both datasets
    results['shootout_id'] = (results['date'].astype(str) + '_' + results['home_team'] + '_' + results['away_team'])
    shootouts['shootout_id'] = (shootouts['date'].astype(str) + '_' + shootouts['home_team'] + '_' + shootouts['away_team'])

    # Covert shootout winner to mapping dict
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
    df['shootout_winner'] == df['away_team']
    ]  

    choices = ['W','L','W','L']
    df['result'] = np.select(match_conditions,choices,default='D')

    return df

######################################## OLD FUNCTIONS (Still in Use) ########################################

def transform_match_data(load_directory: Path = DATA_DIRECTORY) -> tuple[pd.DataFrame,pd.DataFrame]:

    frames = load_raw_data(load_directory)

    results = frames['results']
    shootouts = frames['shootouts']
    former_names = frames['former_names']

    played, not_played = _update_matches_data(results,shootouts,former_names)
    played['year'] = played['date'].dt.year

    match_conditions = [
    played['home_score'] > played['away_score'],
    played['home_score'] < played['away_score'],
    played['shootout_winner'] == played['home_team'],
    played['shootout_winner'] == played['away_team']
    ]  

    choices = ['W','L','W','L']
    played['result'] = np.select(match_conditions,choices,default='D')
    played['goal_diff'] = np.abs(played['home_score'] - played['away_score'])

    _write_processed_data(
        {
            'played':played,
            'not_played':not_played
        }
    )

    return played, not_played

def _write_processed_data(df_dict):

    for key,value in df_dict.items():

        write_path = PROCESSED_DIRECTORY / f'{key}.csv'
        value.to_csv(write_path)

def _update_matches_data(results:pd.DataFrame,shootouts:pd.DataFrame,former_names:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:

    not_played = results[results['home_score'].isna()].copy()
    played_raw = results[~results['home_score'].isna()].copy()

    played_with_shootouts = _enrich_with_shootout(played_raw,shootouts)

    played_final = _update_names(played_with_shootouts,former_names)

    return played_final, not_played

def _enrich_with_shootout(played:pd.DataFrame,shootouts:pd.DataFrame) -> pd.DataFrame:

    played = played.copy()

    played['shootout_id'] = (played['date'].astype(str) + '_' + played['home_team'] + '_' + played['away_team'])
    shootouts['shootout_id'] = (shootouts['date'].astype(str) + '_' + shootouts['home_team'] + '_' + shootouts['away_team'])

    shootout_dict = pd.Series(shootouts['winner'].values, index = shootouts['shootout_id']).to_dict()

    played['shootout_winner'] = played['shootout_id'].map(shootout_dict)
    return played

def _update_names(played:pd.DataFrame,former_names:pd.DataFrame) -> pd.DataFrame:

    played = played.copy()

    name_map = dict(zip(former_names['former'], former_names['current']))

    played['home_team'] = played['home_team'].replace(name_map)
    played['away_team'] = played['away_team'].replace(name_map)

    return played

if __name__ == '__main__':

    df = transform_datasets(PATH)['played']
    