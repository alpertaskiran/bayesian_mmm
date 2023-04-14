import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.tensor.random.utils import params_broadcast_shapes
from sklearn.preprocessing import MaxAbsScaler
from scipy.stats import pearsonr

from utils import delayed_adstock

def plot_posterior(model_posterior_predictive,date_data,target_data,target_scaler,plot_settings):
    posterior_predictive_likelihood = az.extract(
        data=model_posterior_predictive,
        group="posterior_predictive",
        var_names="likelihood",
    )

    posterior_predictive_likelihood_inv = target_scaler.inverse_transform(
        X=posterior_predictive_likelihood
    )

    fig, ax = plt.subplots()

    for i, p in enumerate(plot_settings['percs'][::-1]):
        upper = np.percentile(posterior_predictive_likelihood_inv, p, axis=1)
        lower = np.percentile(posterior_predictive_likelihood_inv, 100 - p, axis=1)
        color_val = plot_settings['colors'][i]
        ax.fill_between(
            x=date_data,
            y1=upper,
            y2=lower,
            color=plot_settings['cmap'](color_val),
            alpha=0.05,
        )

    sns.lineplot(
        x=date_data,
        y=posterior_predictive_likelihood_inv.mean(axis=1),
        color="C2",
        label="posterior predictive mean",
        ax=ax,
    )
    sns.lineplot(
        x=date_data,
        y=target_data,
        color="black",
        label="target (scaled)",
        ax=ax,
    )
    ax.legend(loc="upper left")
    ax.set(title="Base Model - Posterior Predictive Samples")
    plt.show()

def plot_prior_predictive_samples(model_prior_predictive,target_scaled,date_data,plot_settings):
    fig, ax = plt.subplots()

    for i, p in enumerate(plot_settings['percs'][::-1]):
        upper = np.percentile(model_prior_predictive.prior_predictive["likelihood"], p, axis=1)
        lower = np.percentile(
            model_prior_predictive.prior_predictive["likelihood"], 100 - p, axis=1
        )
        color_val = plot_settings['colors'][i]
        ax.fill_between(
            x=date_data,
            y1=upper.flatten(),
            y2=lower.flatten(),
            color=plot_settings['cmap'](color_val),
            alpha=0.1,
        )

    sns.lineplot(x=date_data, y=target_scaled, color="black", label="target (scaled)", ax=ax)
    ax.legend()
    ax.set(title="Model - Prior Predictive Samples")
    plt.show()