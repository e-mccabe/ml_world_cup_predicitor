
from pathlib import Path


def find_root(marker = 'pyproject.toml'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f'No file: {marker} found above {__file__}')

PROJECT_ROOT = find_root()
DATA_DIRECTORY = PROJECT_ROOT / 'data' / 'international-football-results'
PROCESSED_DIRECTORY = PROJECT_ROOT / 'data' / 'processed'

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