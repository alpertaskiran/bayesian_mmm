import pymc as pm
from utils import delayed_adstock


def mmm_model(model_input, target, ll):
    coords = model_input
    max_lag = 13

    with pm.Model(coords=coords) as model:
        # data containers
        channel_data_ = pm.MutableData(
            name='channel_data',
            value=model_input['channel'],
            dims=('date', 'channel'),
        )
        seasonality_ = pm.MutableData(
            name='seasonality_data', value=model_input['seasonality'],
            dims=('date', 'seasonality')
        )
        trend_ = pm.MutableData(
            name='t', value=model_input['trend'], dims=('date')
        )

        target_ = pm.MutableData(name='target', value=target, dims='date')

        # priors
        intercept = pm.Normal(name='intercept', mu=0, sigma=1)
        b_trend = pm.Normal(name='b_trend', mu=0, sigma=1)
        beta_channel = pm.HalfNormal(
            name='beta_channel', sigma=1, dims='channel'
        )
        alpha = pm.Beta(name='alpha', alpha=3, beta=3, dims='channel')
        theta = pm.Uniform('delay', lower=0, upper=max_lag - 1, dims='channel')
        sigma = pm.HalfNormal(name='sigma', sigma=1)
        fourier_control = pm.Laplace(
            name='gamma_control', mu=0, b=1, dims='seasonality'
        )

        # model parametrization
        channel_adstock = pm.Deterministic(
            name='channel_adstock',
            var=delayed_adstock(
                x=channel_data_,
                alpha=alpha,
                theta=theta,
                l_max=max_lag - 1
            ),
            dims=('date', 'channel'),
        )

        channel_contributions = pm.Deterministic(
            name='channel_contributions',
            var=channel_adstock * beta_channel,
            dims=('date', 'channel'),
        )

        mu_var = channel_contributions.sum(axis=-1)

        trend = pm.Deterministic('trend', intercept + b_trend * trend_, dims='date')

        control_contributions = pm.Deterministic(
            name='control_contributions',
            var=seasonality_ * fourier_control,
            dims=('date', 'seasonality'),
        )

        mu_var += control_contributions.sum(axis=-1)
        mu_var += trend
        mu = pm.Deterministic(name='mu', var=mu_var, dims='date')

        if ll == 'Student':
            # degrees of freedom of the t distribution
            nu = pm.Gamma(name='nu', alpha=15, beta=1)

            pm.StudentT(
                name='likelihood',
                nu=nu,
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims='date',
            )
            model_prior_predictive = pm.sample_prior_predictive()

        if ll == 'Normal':
            pm.Normal(
                name='likelihood',
                mu=mu,
                sigma=sigma,
                observed=target_,
                dims='date',
            )
            model_prior_predictive = pm.sample_prior_predictive()
    return model, model_prior_predictive
