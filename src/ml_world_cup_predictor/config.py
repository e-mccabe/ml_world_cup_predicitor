
from pathlib import Path
from dataclasses import dataclass
import datetime as dt

def _find_root(marker = 'pyproject.toml'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f'No file: {marker} found above {__file__}')

PROJECT_ROOT = _find_root()

KAGGLE_LINK = "martj42/international-football-results-from-1872-to-2017"

@dataclass(frozen=True)
class Paths:
    raw:Path        = PROJECT_ROOT / 'data' / 'raw'
    interim:Path    = PROJECT_ROOT / 'data' / 'interim'
    processed:Path  = PROJECT_ROOT / 'data' / 'processed'

PATH = Paths()

DATA_DIRECTORY = PROJECT_ROOT / 'data' / 'international-football-results'
PROCESSED_DIRECTORY = PROJECT_ROOT / 'data' / 'processed'


# Filtering Parameters
MINIMUM_GAMES_PLAYED = 30

WC_START = dt.datetime(2026,6,11)


# ELO Parameters
STARTING_ELO = 1500

GAME_WEIGHTS = {
    'Friendly':20,
    'FIFA World Cup':90,
    'FIFA World Cup qualification':55,
    'Copa América':50,
    'African Cup of Nations':40,
    'UEFA Nations League':50,
    'CONCACAF Nations League':40,
    'UEFA Euro':50,
    'CONCACAF Championship':40
}



# Form Parameters
FORM_WINDOW = 15

