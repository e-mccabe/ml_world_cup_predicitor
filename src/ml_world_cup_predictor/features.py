import pandas as pd
import datetime as dt
from collections import deque

from ml_world_cup_predictor.config import STARTING_ELO,FORM_WINDOW, Paths
from ml_world_cup_predictor.elo import compute_elo_ratings
from ml_world_cup_predictor.form import form_score, result_to_form_points
from ml_world_cup_predictor.transform import transform_datasets

from ml_world_cup_predictor.utils import generate_team_list, load_folder_to_dict

def build_feature_table(path:Paths,features:list[str],label_col:str,refresh:bool = False,date_filter:dt.datetime = None) -> pd.DataFrame:
    """Generates the final feature table for the decision tree, creates the elo and form features from past data"""
    
    if not refresh and any(path.processed.iterdir()):
        processed = load_folder_to_dict(path.processed)['played']
    else:
        processed = transform_datasets(path,refresh)['played']

    final_columns = features + [label_col]

    # If a date filter exists, filter the df to exclude games on or after that date
    if date_filter: 
        processed = processed[processed.date < date_filter]

    # TODO: rework this function using sets and passing the minimum games 
    feature_df = _team_filtering(processed)

    feature_df = _create_feature_columns(feature_df)

    return feature_df[final_columns]


def _team_filtering(processed_df:pd.DataFrame,**filter_kwargs) -> pd.DataFrame:

    processed_df = processed_df.copy()
    teams = generate_team_list(processed_df,**filter_kwargs)
    teams_mask = (processed_df['home_team'].isin(teams)) & (processed_df['away_team'].isin(teams))
    return processed_df[teams_mask].reset_index(drop=True)


def _create_feature_columns(processed_df:pd.DataFrame) -> pd.DataFrame:
    """
    Iterations in date order to generate the features for downstream machine learning
    
    Features:
        > ELO:  pre-match ELO rating
        > FORM: pre-match form points (Win = 3, Draw = 1, Loss = 0)
    """

    teams = set(processed_df['home_team']) | set(processed_df['away_team'])

    current_rating = {team:STARTING_ELO for team in teams}
    recent_results = {team:deque(maxlen = FORM_WINDOW) for team in teams}
    match_history = []

    for row in processed_df.itertuples(index = True,name='match'):

        ### ======= Generating ELO =======
        # Pulling the previous ratings
        home_previous_rating = current_rating[row.home_team]
        away_previous_rating = current_rating[row.away_team]

        # Using ELO equations to calculate updated ratings
        new_home_rating, new_away_rating = compute_elo_ratings(row,home_previous_rating,away_previous_rating)

        # Updating the ratings
        current_rating[row.home_team] = new_home_rating
        current_rating[row.away_team] = new_away_rating

        ### ======= Generating Form =======
        # Pulling the teams previous form
        home_previous_form = form_score(recent_results[row.home_team])
        away_previous_form = form_score(recent_results[row.away_team])

        # Calculating the form in terms of points (W = 3, D = 1, L = 0)
        home_points, away_points = result_to_form_points(row.result)

        # Appending New form
        recent_results[row.home_team].append(home_points)
        recent_results[row.away_team].append(away_points)

        # Generating the features for the current row
        match_history.append({
                        'elo_difference'    : home_previous_rating-away_previous_rating,
                        'abs_elo_difference': abs(home_previous_rating-away_previous_rating),
                        'form_difference'   : home_previous_form - away_previous_form
                        })
        
    match_features = pd.DataFrame(match_history)

    final_df = pd.concat([processed_df,match_features],axis= 1)
    
    return final_df

