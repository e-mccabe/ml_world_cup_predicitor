from collections import deque

def form_score(form_list:deque[int]) -> float:
    """
    Takes the list of the last n results in terms of points and calculates the final score.
    
    Current use case is a simple total of all the points. Keeping the logic in this function allows me to change
    the logic at a later point if required
    """

    return sum(form_list)

def result_to_form_points(result:str) -> tuple[int,int]:
    """Converts the results strings into integers corresponding to the points earned"""

    home_points = 3 if result == 'W' else 0 if result == 'L' else 1
    away_points = 3 if result == 'L' else 0 if result == 'W' else 1

    return home_points, away_points