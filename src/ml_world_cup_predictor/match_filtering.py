import pandas as pd
import datetime as dt
from ml_world_cup_predictor.config import MINIMUM_GAMES_PLAYED

def _generate_team_list(df:pd.DataFrame,minimum_games:int = 0) -> list[str]:
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

def filter_matches_for_rating(df:pd.DataFrame,stop_date:dt.datetime=None,minimum_games:int=MINIMUM_GAMES_PLAYED)-> tuple[pd.DataFrame,list[str]]:
    """Filtering match dataset to specific timeframe and removing teams with minimal games"""
    filtered_by_date = df[df['date'] < stop_date] if stop_date is not None else df

    teams = _generate_team_list(filtered_by_date,minimum_games)
    teams_mask = (filtered_by_date['home_team'].isin(teams)) & (filtered_by_date['away_team'].isin(teams)) 

    return filtered_by_date[teams_mask].reset_index(drop=True), teams