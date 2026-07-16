from ml_world_cup_predictor.feature_engineering import generate_match_features
from ml_world_cup_predictor.data_transformation import transform_match_data 
from ml_world_cup_predictor.config import DATA_DIRECTORY

def main():

    print("Hello from ml-world-cup-predictor!")
    played,not_played = transform_match_data(DATA_DIRECTORY)
    df = generate_match_features(played)
    print(df.tail(5))

if __name__ == "__main__":
    main()
