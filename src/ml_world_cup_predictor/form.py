import numpy as np
from collections import deque

def _form_score(form_list:deque[int]) -> float:

    return sum(form_list)#/((len(form_list)) if form_list else np.nan)

def _result_to_form_points(result:str) -> tuple[int,int]:

        home_points = 3 if result == 'W' else 0 if result == 'L' else 1
        away_points = 3 if result == 'L' else 0 if result == 'W' else 1

        return home_points, away_points