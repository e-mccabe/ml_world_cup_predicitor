from typing import Any

from ml_world_cup_predictor.config import GAME_WEIGHTS, TOURNAMENT_FALLBACK, HOME_ADVANTAGE_DEFAULT

def _result_to_elo_outcome(result:str)-> tuple[float,float]:
    """Mapping result string to corresponding outcome tuple (home team, away team)"""
    if result == 'W':
        return 1,0
    elif result == 'L':
        return 0,1
    else:
        return 0.5,0.5

def elo_expected_outcome(home_rating:float,away_rating:float,c:float =10,scaling_constant:float=400,home_advantage:float= HOME_ADVANTAGE_DEFAULT) -> float:
    """
    ELO expected outcome formula, calculates which team is expected to win using the previous ratings of the teams

    parameters:
        > home_rating       : current ELO rating of the home team
        > away_rating       : current ELO rating of the away team
        > c                 : scaling parameter; 10 allows rating differences to correspond to magnitudes of 10
        > scaling constant  : used for the difference in standard of team, a 400 point difference implies a team
                              is roughly 10 times more likely to win
        > home advantage    : bonus points added to the home team's rating to capture home advantage
    """
    exponent = (away_rating - home_rating - home_advantage)/scaling_constant

    return 1/(1+c**(exponent))


def elo_rank_update(previous_rating:float,goal_difference:float,actual_outcome:float,expected_outcome:float,match_importance_index:float =50) -> float:
    """
    Computes the change to the ELO rating for a single team based on the outcome of the game
    
    parameters:
        > previous_rating           : the team's elo rating before the game,
        > goal_difference           : absolute value of the goal difference, cannot be negative as it is a multiplier
                                      (the negative sign for the loser's equation comes from the outcome terms since expected outcome is always between 0 and 1).
        > actual_outcome            : actual result of the match (Win = 1, Draw = 0.5, Loss = 0)
        > expected_outcome          : the predicted outcome of the match calculated using the elo_expected_outcome function.
        > match_importance_index    : represents the maximum number of points a team can win/lose in a match. Bigger tournaments can be weighted higher.
    
    """

    importance_factor = match_importance_index * (1+goal_difference)
    
    return previous_rating + importance_factor * (actual_outcome - expected_outcome)


def compute_elo_ratings(match:Any,home_previous_rating:float,away_previous_rating:float) -> tuple[float,float]:
    """
    For each row/match calculate the updated elo ratings for the home and away teams
    
    Each match/row must contain:
        > .neutral (bool)       : True if the match is neutral, False if not
        > .tournament (str)     : The name of the tournament the match is played in
        > .goal_diff (float)    : The absolute value of home_score - away_score
        > .result (str)         : The outcome of the match (W = Home Win, D = Draw, L = Home Loss/Away Win)
    
    """
    home_advantage = HOME_ADVANTAGE_DEFAULT if not match.neutral else 0

    # Updating the match importance index by tournament
    match_importance_index = GAME_WEIGHTS.get(match.tournament,TOURNAMENT_FALLBACK)

    home_expected = elo_expected_outcome(home_previous_rating,away_previous_rating,home_advantage = home_advantage)
    away_expected = 1 - home_expected

    goal_difference = match.goal_diff

    home_outcome, away_outcome = _result_to_elo_outcome(match.result)

    new_home_rating = elo_rank_update(home_previous_rating,goal_difference,home_outcome,home_expected,match_importance_index=match_importance_index)
    new_away_rating = elo_rank_update(away_previous_rating,goal_difference,away_outcome,away_expected,match_importance_index=match_importance_index)

    return new_home_rating, new_away_rating