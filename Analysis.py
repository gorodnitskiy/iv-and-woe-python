from typing import List, Tuple
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from Features import ContinuousFeature, CategoricalFeature
from AttributeRelevance import IV, AttributeRelevance


def feat_values(
    df: pd.DataFrame,
    col_list: List[str],
    label: str
) -> List[any]:
    feats_dict = {}
    for col in col_list:
        if is_numeric_dtype(df[col]):
            feats_dict[col] = ContinuousFeature(
                df, col, label, bin_min_size=0.05)
        else:
            feats_dict[col] = CategoricalFeature(
                df, col, label)

    return list(feats_dict.values())


def date_to_quarters(
    df: pd.DataFrame,
    date_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    df['quarter'] = pd.to_datetime(df[date_col])
    df['quarter'] = df['quarter'].apply(
        lambda date: "{}-{}".format(date.year, date.quarter))
    return df, sorted(list(df['quarter'].unique()))


def roc_auc_by_col(
    df: pd.DataFrame,
    col_list: List[str],
    label: str
) -> pd.DataFrame:
    result = []
    for col in col_list:
        if df[label].nunique() == 1:
            roc_auc = np.nan
        else:
            roc_auc = roc_auc_score(df[label], df[col])
        result.append(roc_auc)

    return pd.DataFrame(result, index=col_list, columns=['roc_auc'])


def analysis_by_quarter(
    df: pd.DataFrame,
    col_list: List[str],
    iv: IV,
    ar: AttributeRelevance,
    date_col: str,
    label: str,
    add_label_info: bool = True
) -> pd.DataFrame:
    df, quarters = date_to_quarters(df=df, date_col=date_col)
    feats = feat_values(df, col_list, label=label)
    cond = df['quarter'] == quarters[0]
    result = pd.concat([
        ar.bulk_iv(feats, iv, cond),
        roc_auc_by_col(df[cond], col_list, label)],
        axis=1)
    for quarter in quarters[1:]:
        cond = df['quarter'] == quarter
        result_i = pd.concat([
            ar.bulk_iv(feats, iv, cond),
            roc_auc_by_col(df[df['quarter'] == quarter], col_list, label)],
            axis=1)
        result = pd.concat([result, result_i], axis=1)

    features_nm = list(result.index)
    values_nm = ["IV", "ROC_AUC"]
    result = result.values.tolist()
    result = np.array(result) \
        .reshape(len(features_nm), len(quarters) * len(values_nm))
    multi_columns = pd.MultiIndex.from_product([quarters, values_nm])
    result = pd.DataFrame(result, index=features_nm, columns=multi_columns)

    if add_label_info:
        mean_label = df[['quarter', label]] \
            .groupby(['quarter']) \
            .agg({label: ['mean', 'count']}) \
            .sort_index() \
            .values \
            .tolist()

        mean_label_mean, mean_label_count = \
            [value[0] for value in mean_label for _ in range(0, 2)], \
            [value[1] for value in mean_label for _ in range(0, 2)]
        mean_label = [mean_label_mean, mean_label_count]
        mean_label = np.array(mean_label) \
            .reshape(2, len(quarters) * len(values_nm))
        mean_label = pd.DataFrame(
            mean_label,
            index=['Target by quarter', 'Count by target'],
            columns=multi_columns)

        result = pd.concat([result, mean_label], axis=0)

    for col in result.columns:
        result[col] = result[col].apply(lambda x: round(x, 2))

    return result.T
