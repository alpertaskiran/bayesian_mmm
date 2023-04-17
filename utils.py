import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor.random.utils import params_broadcast_shapes

def summary_table(raw_data_train):
    summary_table = pd.DataFrame(index =raw_data_train.columns)
    summary_table['dtypes'] = raw_data_train.dtypes
    summary_table['unique_values'] = raw_data_train.apply(lambda col: len(col.unique()))
    summary_table['pct_unique_value']= summary_table['unique_values'] / raw_data_train.shape[0]
    summary_table['nan_values'] = raw_data_train.apply(lambda col: col.isna().sum())
    stats=raw_data_train.describe(include='all',datetime_is_numeric=True).T
    required_columns= ['min','max','mean','std']
    summary_table[required_columns]=stats[required_columns]

    return summary_table

def _batched_convolution(x, w, axis: int = 0):
    # used from https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py
    orig_ndim = x.ndim
    axis = axis if axis >= 0 else orig_ndim + axis
    w = pt.as_tensor(w)
    x = pt.moveaxis(x, axis, -1)
    l_max = w.type.shape[-1]
   
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
    # used from https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py
  
    w = pt.power(
        pt.as_tensor(alpha)[..., None],
        (pt.arange(l_max, dtype=x.dtype) - pt.as_tensor(theta)[..., None]) ** 2,
    )
    w = w / pt.sum(w, axis=-1, keepdims=True)
    return _batched_convolution(x, w, axis=axis)
