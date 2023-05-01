import pandas as pd
import pytensor.tensor as pt
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
from pytensor.tensor.random.utils import params_broadcast_shapes


def summary_table(raw_data_train):
    summary_table = pd.DataFrame(index=raw_data_train.columns)
    summary_table['dtypes'] = raw_data_train.dtypes
    summary_table['unique_values'] = raw_data_train.apply(lambda col: len(col.unique()))
    summary_table['pct_unique_value'] = summary_table['unique_values'] / \
        raw_data_train.shape[0]
    summary_table['nan_values'] = raw_data_train.apply(lambda col: col.isna().sum())
    stats = raw_data_train.describe(include='all', datetime_is_numeric=True).T
    required_columns = ['min', 'max', 'mean', 'std']
    summary_table[required_columns] = stats[required_columns]

    return summary_table


def _batched_convolution(x, w, axis: int = 0):
    # amazingly efficient calculation is borrowed from :
    #  https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py
    orig_ndim = x.ndim
    axis = axis if axis >= 0 else orig_ndim + axis
    w = pt.as_tensor(w)
    x = pt.moveaxis(x, axis, -1)
    l_max = w.type.shape[-1]
    if l_max is None:
        try:
            l_max = w.shape[-1].eval()
        except Exception:
            pass
    x_shape, w_shape = params_broadcast_shapes([x.shape, w.shape], [1, 1])
    x = pt.broadcast_to(x, x_shape)
    w = pt.broadcast_to(w, w_shape)
    x_time = x.shape[-1]
    shape = (*x.shape, w.shape[-1])
    padded_x = pt.zeros(shape, dtype=x.dtype)
    for i in range(l_max):
        padded_x = pt.set_subtensor(
            padded_x[..., i:x_time, i], x[..., : x_time - i]
        )
    conv = pt.sum(padded_x * w[..., None, :], axis=-1)
    return pt.moveaxis(conv, -1, axis + conv.ndim - orig_ndim)


def delayed_adstock(
    x,
    alpha: float = 0.0,
    theta: int = 0,
    l_max: int = 12,
    axis: int = 0,
):
    # amazingly efficient calculation is borrowed from :
    #  https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py
    w = pt.power(
        pt.as_tensor(alpha)[..., None],
        (pt.arange(l_max, dtype=x.dtype) - pt.as_tensor(theta)[..., None]) ** 2,
    )
    w = w / pt.sum(w, axis=-1, keepdims=True)
    return _batched_convolution(x, w, axis=axis)


def max_abs_scaler(df, boolean_target):
    if boolean_target:
        scaler = MaxAbsScaler()
        scaler.fit(df.values.reshape(-1, 1))
        scaled_df = pd.Series(scaler.transform(df.values.reshape(-1, 1)).flatten())
    else:
        scaler = MaxAbsScaler()
        scaler.fit(df)
        scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return scaler, scaled_df


def create_trend_seasonality(df):
    # creating trend and seasonality
    n_order = 7
    periods = df['day'] / 365.25
    fourier_features = pd.DataFrame(
        {
            f'{func}_order_{order}': getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ('sin', 'cos')
        }
    )
    trend = (df.index - df.index.min()) / (df.index.max() - df.index.min())
    seasonality = fourier_features

    return pd.Series(data=trend, name='trend'), seasonality
