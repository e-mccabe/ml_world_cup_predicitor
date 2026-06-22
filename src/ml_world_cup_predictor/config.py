
from pathlib import Path

DATA_DIRECTORY = Path("../data/international-football-results")

PROCESSED_DIRECTORY = Path("/data/processed_data")

STARTING_ELO = 1500

FORM_WINDOW = 5

MINIMUM_GAMES_PLAYED = 30

GAME_WEIGHTS = {
    'Friendly':20,
    'FIFA World Cup':60,
    'FIFA World Cup qualification':40,
    'Copa América':50,
    'African Cup of Nations':50,
    'UEFA Nations League':50,
    'CONCACAF Nations League':50,
    'UEFA Euro':50,
    'CONCACAF Championship':50
}