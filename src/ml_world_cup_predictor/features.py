import pandas as pd
import datetime as dt
from pathlib import Path
from collections import deque

from ml_world_cup_predictor.config import STARTING_ELO,FORM_WINDOW,PROCESSED_DIRECTORY,DATA_DIRECTORY,PATH
from ml_world_cup_predictor.match_filtering import filter_matches_for_rating
from ml_world_cup_predictor.elo import compute_elo_ratings
from ml_world_cup_predictor.form import _form_score, _result_to_form_points
from ml_world_cup_predictor.transform import transform_match_data, transform_datasets

from ml_world_cup_predictor.utils import generate_team_list, load_folder_to_dict


def build_feature_table(path:Path,features:list[str],label_col:str,refresh:bool = False,**filter_kwargs) -> pd.DataFrame:

    if not refresh and any(path.processed.iterdir()):
        processed = load_folder_to_dict(path.processed)['played']
    else:
        processed = transform_datasets(path,refresh)['played']

    columns = features + [label_col]

    feature_df = _team_filtering(processed)

    feature_df = _create_feature_columns(feature_df)

    if filter_kwargs.get('date',None):
        feature_df = feature_df[feature_df.date < filter_kwargs['date']]

    return feature_df[columns]


def _team_filtering(processed_df:pd.DataFrame,**filter_kwargs) -> pd.DataFrame:

    processed_df = processed_df.copy()
    teams = generate_team_list(processed_df,**filter_kwargs)
    teams_mask = (processed_df['home_team'].isin(teams)) & (processed_df['away_team'].isin(teams))
    return processed_df[teams_mask].reset_index(drop=True)

def _create_feature_columns(processed_df:pd.DataFrame):

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
        home_previous_form = _form_score(recent_results[row.home_team])
        away_previous_form = _form_score(recent_results[row.away_team])

        # Calculating the form in terms of points (W = 3, D = 1, L = 0)
        home_points, away_points = _result_to_form_points(row.result)

        # Appending New form
        recent_results[row.home_team].append(home_points)
        recent_results[row.away_team].append(away_points)

        # Generating the features for the current row
        match_history.append({
                        'elo_difference'    : home_previous_rating-away_previous_rating,
                        'form_difference'   : home_previous_form - away_previous_form
                        })
        
    match_features = pd.DataFrame(match_history)

    final_df = pd.concat([processed_df,match_features],axis= 1)
    
    return final_df


######################################## OLD FUNCTIONS (Still in Use) ########################################

def get_data_for_tree_build(features:list[str],y_col:str,**filter_kwargs) -> pd.DataFrame:
    features.append(y_col)
    processed = _load_processed_data()
    updated_with_features = generate_match_features(processed['played'],**filter_kwargs)
    return updated_with_features[features]

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
                        'home_elo'          : home_previous_rating,
                        'away_elo'          : away_previous_rating,
                        'elo_difference'    : home_previous_rating-away_previous_rating,
                        'home_form'         : home_previous_form,
                        'away_form'         : away_previous_form,
                        'form_difference'   : home_previous_form - away_previous_form
                        })

    match_features = pd.DataFrame(match_history)

    matches_filtered = pd.concat([matches_filtered,match_features],axis= 1)
    
    return matches_filtered

def _load_processed_data():

    if not any(PROCESSED_DIRECTORY.iterdir()):
        transform_match_data(DATA_DIRECTORY)
    return {file_path.stem : pd.read_csv(file_path) for file_path in sorted(PROCESSED_DIRECTORY.glob('*csv'))}

if __name__ == '__main__':

    df = build_feature_table(PATH,['elo_difference','form_difference'],'result')
    
    
