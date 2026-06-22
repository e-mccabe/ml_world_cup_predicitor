import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from collections import deque
from typing import Any
from ml_world_cup_predictor.config import STARTING_ELO , MINIMUM_GAMES_PLAYED, GAME_WEIGHTS, PROCESSED_DIRECTORY, FORM_WINDOW

def _result_to_elo_outcome(result:str)-> tuple[float,float]:
        if result == 'W':
            return 1,0
        elif result == 'L':
            return 0,1
        else:
            return 0.5,0.5

def _generate_team_list(df:pd.DataFrame,minimum_games:int = 0) -> list[str]:

    df =df.copy()
    
    df = df.melt(
    id_vars=['date','year','tournament'],
    value_vars=['home_team','away_team'],
    value_name='team'
)
    count = df['team'].value_counts()
    count = count[count >= minimum_games]

    return sorted(list(count.index)) 

def _form_score(form_list:deque[int]) -> float:

    return sum(form_list)/((3*len(form_list)) if form_list else np.nan)

def elo_expected_outcome(home_rating:float,away_rating:float,c:float =10,d:float=400,omega:float= 50) -> float:

    exponent = (away_rating - home_rating - omega)/d

    return 1/(1+c**(exponent))

def elo_rank_update(previous_rating:float,goal_difference:float,actual_outcome:float,expected_outcome:float,k0:float =50) -> float:

    importance_factor = k0 * (1+goal_difference)
    
    return previous_rating + importance_factor * (actual_outcome - expected_outcome)


def filter_matches_for_elo(df:pd.DataFrame,stop_date:dt.datetime=None,minimum_games:int=MINIMUM_GAMES_PLAYED)-> tuple[pd.DataFrame,list[str]]:

    filtered_by_date = df[df['date'] < stop_date] if stop_date is not None else df

    teams = _generate_team_list(filtered_by_date,minimum_games)
    teams_mask = (filtered_by_date['home_team'].isin(teams)) & (filtered_by_date['away_team'].isin(teams)) 

    return filtered_by_date[teams_mask].reset_index(drop=True), teams


def compute_elo_ratings(row:Any,home_previous_rating:float,away_previous_rating:float) -> tuple[float,float]:

    omega = 100 if not row.neutral else 0
    k0 = GAME_WEIGHTS.get(row.tournament,30)

    home_expected = elo_expected_outcome(home_previous_rating,away_previous_rating,omega = omega)
    away_expected = 1 - home_expected

    goal_difference = row.goal_diff

    home_outcome, away_outcome = _result_to_elo_outcome(row.result)

    new_home_rating = elo_rank_update(home_previous_rating,goal_difference,home_outcome,home_expected,k0=k0)
    new_away_rating = elo_rank_update(away_previous_rating,goal_difference,away_outcome,away_expected,k0=k0)

    return new_home_rating, new_away_rating

def _result_to_form_points(result:str) -> tuple[int,int]:

        home_points = 3 if result == 'W' else 0 if result == 'L' else 1
        away_points = 3 if result == 'L' else 0 if result == 'W' else 1

        return home_points, away_points

def generate_match_features(matches_df:pd.DataFrame,stop_date:dt.datetime = None,write_path:Path = PROCESSED_DIRECTORY) -> pd.DataFrame:

    matches_filtered, teams = filter_matches_for_elo(matches_df,stop_date) 

    current_rating = {team : STARTING_ELO for team in teams}
    recent_results = {team:deque(maxlen = FORM_WINDOW) for team in teams}
    match_history = []

    for row in matches_filtered.itertuples(index = True,name='match'):
        home_previous_rating = current_rating[row.home_team]
        away_previous_rating = current_rating[row.away_team]

        home_previous_form = _form_score(recent_results[row.home_team])
        away_previous_form = _form_score(recent_results[row.away_team])

        new_home_rating, new_away_rating = compute_elo_ratings(row,home_previous_rating,away_previous_rating)

        home_points, away_points = _result_to_form_points(row.result)
        
        recent_results[row.home_team].append(home_points)
        recent_results[row.away_team].append(away_points)

        current_rating[row.home_team] = new_home_rating
        current_rating[row.away_team] = new_away_rating


        match_history.append({
                        'home_elo'  : home_previous_rating,
                        'away_elo'  : away_previous_rating,
                        'home_form' : home_previous_form,
                        'away_form' : away_previous_form
                        })

    match_features = pd.DataFrame(match_history)

    matches_filtered = pd.concat([matches_filtered,match_features],axis= 1)
    
    write_path.mkdir(parents=True,exist_ok=True)
    file_path = write_path / 'elo.csv'
    matches_filtered.to_csv(file_path,index=True)

    return matches_filtered

