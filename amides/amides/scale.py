"""This module contains functions to create symmetric output value scalers."""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef


def create_min_max_scaler(df_min: float, df_max: float):
    """Create symmetric min-max scaler.

    Parameters
    ----------
    df_min: float
        The minimum df-value

    df_max: float
        The maximum df_value

    Returns
    -------
    : MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.data_min_ = df_min
    scaler.data_max_ = df_max

    data_range = df_max - df_min
    scaler.data_range_ = data_range

    scale = 1 / data_range
    scaler.scale_ = scale
    scaler.min_ = 0 - df_min * scale

    return scaler


def create_symmetric_mcc_min_max_scaler(
    df_values: np.ndarray,
    labels: np.ndarray,
    num_mcc_samples: int,
    mcc_threshold: float,
):
    """Creates a symmetric min-max scaler using mcc threshold optimization."""
    df_min, df_max = df_values.min(), df_values.max()
    iter_values = _calculate_iter_values(df_min, df_max, num_mcc_samples)
    mcc = _calculate_mcc_values(df_values, labels, iter_values)

    target_df_values = _calculate_target_df_values(mcc, iter_values, mcc_threshold)
    target_df_min, target_df_max = _calculate_symmetric_min_max_df_values(
        target_df_values
    )

    # Repeat process of MCC optimization after target df-value range was calculated
    # in order to increase precision
    target_iter_values = _calculate_iter_values(
        target_df_min, target_df_max, num_mcc_samples
    )
    target_mcc = _calculate_mcc_values(df_values, labels, target_iter_values)
    target_df_values = _calculate_target_df_values(
        target_mcc, target_iter_values, mcc_threshold
    )
    target_df_min, target_df_max = _calculate_symmetric_min_max_df_values(
        target_df_values
    )

    return create_min_max_scaler(target_df_min, target_df_max)


def create_symmetric_min_max_scaler(df_values: np.ndarray):
    """Create symmetric min-max scaler."""
    df_min, df_max = _calculate_symmetric_min_max_df_values(df_values)

    return create_min_max_scaler(df_min, df_max)


def _calculate_iter_values(min_value: float, max_value: float, num_iter_values: int):
    iter_step = (max_value - min_value) / num_iter_values

    return np.arange(min_value, max_value + iter_step, iter_step)


def _calculate_target_df_values(
    mcc: np.ndarray, df_iter_values: np.ndarray, mcc_threshold: float
):
    target_idcs = np.where(mcc > mcc_threshold)[0]

    return df_iter_values[target_idcs]


def _calculate_mcc_values(df_values: np.ndarray, labels: np.ndarray, iter_values: int):
    mcc = np.zeros(shape=(iter_values.size,))
    for i, threshold in enumerate(iter_values):
        predict = np.where(df_values >= threshold, 1, 0)
        mcc[i] = matthews_corrcoef(y_true=labels, y_pred=predict)

    return mcc


def _calculate_symmetric_min_max_df_values(df_values: np.ndarray):
    df_min = df_values.min()
    df_max = df_values.max()

    if abs(df_min) > df_max:
        df_max = abs(df_min)
    elif df_max > abs(df_min):
        df_min = df_max * -1.0

    return df_min, df_max
