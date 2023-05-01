import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


def plot_posterior(model_posterior_predictive,
                   date_data, target_data,
                   target_scaler, plot_settings):
    posterior_predictive_ll = az.extract(
        data=model_posterior_predictive,
        group='posterior_predictive',
        var_names='likelihood',
    )

    posterior_predictive_ll_inv = target_scaler.inverse_transform(
        X=posterior_predictive_ll
    )

    fig, ax = plt.subplots()

    for i, p in enumerate(plot_settings['percs'][::-1]):
        upper = np.percentile(posterior_predictive_ll_inv, p, axis=1)
        lower = np.percentile(posterior_predictive_ll_inv, 100 - p, axis=1)
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
        y=posterior_predictive_ll_inv.mean(axis=1),
        color='C2',
        label='posterior predictive mean',
        ax=ax,
    )
    sns.lineplot(
        x=date_data,
        y=target_data,
        color='black',
        label='revenue)',
        ax=ax,
    )
    ax.legend(loc='upper left')
    ax.set(title='Posterior Predictive Samples')
    plt.xticks(rotation=30)
    plt.show()


def plot_prior_predictive_samples(model_prior_predictive,
                                  target_scaled, date_data,
                                  plot_settings):
    fig, ax = plt.subplots()

    for i, p in enumerate(plot_settings['percs'][::-1]):
        upper = np.percentile(
            model_prior_predictive.prior_predictive['likelihood'], p, axis=1)
        lower = np.percentile(
            model_prior_predictive.prior_predictive['likelihood'], 100 - p, axis=1
        )
        color_val = plot_settings['colors'][i]
        ax.fill_between(
            x=date_data,
            y1=upper.flatten(),
            y2=lower.flatten(),
            color=plot_settings['cmap'](color_val),
            alpha=0.1,
        )

    sns.lineplot(x=date_data, y=target_scaled, color='black',
                 label='revenue(scaled)', ax=ax)
    ax.legend()
    ax.set(title='Model - Prior Predictive Samples')
    plt.xticks(rotation=30)
    plt.show()
