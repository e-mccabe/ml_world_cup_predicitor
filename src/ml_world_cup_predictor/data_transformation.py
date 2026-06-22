import pandas as pd
import numpy as np
from pathlib import Path
from ml_world_cup_predictor.config import DATA_DIRECTORY
from ml_world_cup_predictor.data_loading import load_raw_data


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

    return played, not_played



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

