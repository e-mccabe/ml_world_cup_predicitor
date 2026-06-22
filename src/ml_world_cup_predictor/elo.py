from typing import Any

from ml_world_cup_predictor.config import GAME_WEIGHTS

def _result_to_elo_outcome(result:str)-> tuple[float,float]:
        if result == 'W':
            return 1,0
        elif result == 'L':
            return 0,1
        else:
            return 0.5,0.5

def elo_expected_outcome(home_rating:float,away_rating:float,c:float =10,d:float=400,omega:float= 50) -> float:

    exponent = (away_rating - home_rating - omega)/d

    return 1/(1+c**(exponent))


def elo_rank_update(previous_rating:float,goal_difference:float,actual_outcome:float,expected_outcome:float,k0:float =50) -> float:

    importance_factor = k0 * (1+goal_difference)
    
    return previous_rating + importance_factor * (actual_outcome - expected_outcome)


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