import numpy as np
import pandas as pd

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

    print(f'Accuracy = {100*(correctly_predicted.sum()/row_sum.sum()):.1f}%')

    recall = {
        column:predicted/actual  if actual > 0 else 0 
        for column, predicted, actual in zip(confusion_matrix.columns,correctly_predicted,row_sum)}
    
    precision = {
        column:predicted/actual  if actual > 0 else 0 
        for column, predicted, actual in zip(confusion_matrix.columns,correctly_predicted,column_sum)}

    return pd.DataFrame([recall,precision],index = ['recall','prediction'])
    