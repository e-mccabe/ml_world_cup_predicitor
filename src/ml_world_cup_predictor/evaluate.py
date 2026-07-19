import numpy as np
import pandas as pd
from ml_world_cup_predictor.logging import ModelMetrics,PerClassMetrics

def confusion_matrix(true_labels:pd.Series,predicted_labels:list)-> pd.DataFrame:

    predictions = pd.Series(predicted_labels, index = true_labels.index)
    confusion_matrix = pd.crosstab(true_labels,predictions)

    # Ensuring that the matrix is square and no labels are dropped if not predicted
    all_labels = confusion_matrix.index.union(confusion_matrix.columns)
    confusion_matrix = confusion_matrix.reindex(index = all_labels,columns=all_labels,fill_value= 0)
    
    return confusion_matrix


def classification_summary(confusion_matrix:pd.DataFrame):

    correctly_predicted = np.diag(confusion_matrix) 

    row_sum = confusion_matrix.sum(axis=1)
    column_sum = confusion_matrix.sum(axis=0)

    accuracy = (correctly_predicted.sum()/row_sum.sum())

    class_recall = {
        column:predicted/actual  if actual > 0 else 0 
        for column, predicted, actual in zip(confusion_matrix.columns,correctly_predicted,row_sum)}
    
    class_precision = {
        column:predicted/actual  if actual > 0 else 0 
        for column, predicted, actual in zip(confusion_matrix.columns,correctly_predicted,column_sum)}

    class_metrics = PerClassMetrics(
        recall=class_recall,
        precision=class_precision
    )

    model_metrics = ModelMetrics(
        accuracy=accuracy,
        class_metrics=class_metrics
    )

    return pd.DataFrame([class_recall,class_precision],index = ['recall','precision']),model_metrics