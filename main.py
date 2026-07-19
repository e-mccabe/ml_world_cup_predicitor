import time
import numpy as np

from ml_world_cup_predictor.model import DecisionTree
from ml_world_cup_predictor.features import build_feature_table
from ml_world_cup_predictor.utils import chronological_split
from ml_world_cup_predictor.config import DATA_PATH, WORLD_CUP_START
from ml_world_cup_predictor.evaluate import confusion_matrix, classification_summary
from ml_world_cup_predictor.logging import write_run

FEATURES = ['elo_difference','form_difference','abs_elo_difference','draw_pct']


def evaluate_model():
    """Train on the pre-tournament history, evaluate on the holdout split and log the run"""
    start_time = time.time()

    feature_df = build_feature_table(DATA_PATH, FEATURES, 'result', date_filter=WORLD_CUP_START)
    train, test = chronological_split(feature_df)

    decision_tree = DecisionTree(max_depth=6,class_weight='balanced')
    decision_tree.fit(train, 'result')
    predictions = decision_tree.predict(test)

    cf = confusion_matrix(test['result'], predictions)
    print(cf)

    summary, model_metrics = classification_summary(cf)
    print(summary)

    run_time = f'{time.time() - start_time}s'
    try:
        write_run(run_time, model_metrics)
    except Exception as e:
        print(f'Error Writing Logs: {e}')



def predict_fixtures():
    """Fit on the full played history and predict the remaining fixtures"""
    feature_df = build_feature_table(DATA_PATH, FEATURES, 'result', mode='predict')

    played = feature_df[feature_df['result'].notna()]
    fixtures = feature_df[feature_df['result'].isna()].copy()

    decision_tree = DecisionTree(max_depth=6, class_weight='balanced')
    decision_tree.fit(played[FEATURES + ['result']], 'result')

    fixtures['prediction'] = decision_tree.predict(fixtures[FEATURES])

    return fixtures


def main():
    evaluate_model()
    print(predict_fixtures().to_string())


if __name__ == "__main__":
    main()
