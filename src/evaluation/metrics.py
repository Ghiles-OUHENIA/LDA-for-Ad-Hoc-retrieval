from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np


def calculate_average_precision(
    true_relevance,
    models_predictions,
    models_names=["QL", "CBDM", "LBDM"],
    return_df=True,
):
    result = np.zeros(len(models_predictions))

    for m in range(len(models_predictions)):
        somme = 0

        for q in range(len(true_relevance)):
            somme += average_precision_score(
                true_relevance[q], models_predictions[m, q]
            )

        somme /= len(true_relevance)
        result[m] = somme

    if return_df:
        df = pd.DataFrame({"Algorithme": models_names, "Average Precision": result})
        return df

    return result
