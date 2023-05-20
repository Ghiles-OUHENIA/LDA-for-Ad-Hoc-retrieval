from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np


def calculate_average_precision(
    y_true_list, y_scores_list, num_thresholds=11, column_names=["QL", "CBDM", "LBDM"]
):
    num_queries = len(y_true_list)
    average_precisions = np.zeros((len(y_scores_list), num_queries, num_thresholds))

    for model_index in range(len(y_scores_list)):
        for query_index in range(num_queries):
            y_true = y_true_list[query_index]
            y_scores = y_scores_list[model_index, query_index]

            precision, recall, _ = precision_recall_curve(y_true, y_scores)

            for threshold_index, recall_threshold in enumerate(
                np.linspace(0, 1, num=num_thresholds)
            ):
                mask = recall >= recall_threshold
                average_precision = precision[mask].mean()
                average_precisions[
                    model_index, query_index, threshold_index
                ] = average_precision

    average_precisions_mean = np.zeros((len(y_scores_list), num_thresholds))
    for model_index in range(len(y_scores_list)):
        average_precisions_mean[model_index] = average_precisions[model_index].mean(
            axis=0
        )

    df = pd.DataFrame(average_precisions_mean.T, columns=column_names)

    return df
