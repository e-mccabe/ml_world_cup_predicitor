import pandas as pd
import datetime as dt
from pathlib import Path
from collections import deque

from ml_world_cup_predictor.config import STARTING_ELO,FORM_WINDOW
from ml_world_cup_predictor.match_filtering import filter_matches_for_rating
from ml_world_cup_predictor.elo import compute_elo_ratings
from ml_world_cup_predictor.form import _form_score, _result_to_form_points


def generate_match_features(matches_df:pd.DataFrame,stop_date:dt.datetime = None) -> pd.DataFrame:

    matches_filtered, teams = filter_matches_for_rating(matches_df,stop_date) 

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

    return matches_filtered
