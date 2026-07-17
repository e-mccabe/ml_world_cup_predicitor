from ml_world_cup_predictor.model import DecisionTree
from ml_world_cup_predictor.features import build_feature_table
from ml_world_cup_predictor.utils import chronological_split
from ml_world_cup_predictor.config import DATA_PATH, WORLD_CUP_START

def main():

    feature_df = build_feature_table(DATA_PATH,['elo_difference','form_difference'],'result',date_filter=WORLD_CUP_START)
    train, test = chronological_split(feature_df)

    decision_tree = DecisionTree(max_depth=5)

    decision_tree.fit(train,'result')
    predictions = decision_tree.predict(test)

    print(len(predictions),len(train))

    return predictions
if __name__ == "__main__":
    main()
