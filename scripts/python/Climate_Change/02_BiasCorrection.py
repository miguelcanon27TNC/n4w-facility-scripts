import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from scipy.interpolate import interp1d
from functools import partial
from multiprocessing import Pool

# Set an environment variable for compatibility with geopandas
os.environ['USE_PYGEOS'] = '0'


def bias_correction(df_obs, p, s, method='delta', factor_Bcsd=None, nbins=10, extrapolate=None):
    """
    Bias correction techniques for climate model outputs.

    Methods available:
    - 'delta': Add the mean change signal to the observations. Suitable for variables without range limits.
    - 'scaling_add': Additive scaling between the mean of observations and simulations during the training period.
    - 'scaling_multi': Multiplicative scaling for variables.
    - 'eqm': Empirical Quantile Mapping to adjust the cumulative distribution function (CDF).
    - 'BCSD': A specific bias correction method based on monthly factors.

    Parameters:
    - df_obs: Observed climate data for the training period.
    - p: Climate simulated by the model for the same variable and period as df_obs (historical).
    - s: Simulated climate for the projection period.
    - method: Bias correction method ('delta', 'scaling_add', 'scaling_multi', 'eqm', 'BCSD').
    - factor_Bcsd: Correction factor for the BCSD method.
    - nbins: Number of quantiles for 'eqm' (default=10).
    - extrapolate: Extrapolation method for values outside the quantile range ('constant' or None).

    Returns:
    - c: Bias-corrected series for s.
    """

    obs = df_obs.values  # Extract observed values

    # Empirical Quantile Mapping (eqm) method
    if (method == 'eqm') and (nbins > 1):
        binmid = np.arange((1. / nbins) * 0.5, 1., 1. / nbins)  # Calculate bin midpoints
        qo = mquantiles(obs[np.isfinite(obs)], prob=binmid)  # Quantiles of observed data
        qp = mquantiles(p[np.isfinite(p)], prob=binmid)  # Quantiles of simulated historical data
        p2o = interp1d(qp, qo, kind='linear', bounds_error=False)  # Interpolation from simulated to observed quantiles
        c = p2o(s)  # Apply interpolation to projection data

        # Handle extrapolation if necessary
        if extrapolate is None:
            c[s > np.max(qp)] = qo[-1]
            c[s < np.min(qp)] = qo[0]
        elif extrapolate == 'constant':
            c[s > np.max(qp)] = s[s > np.max(qp)] + qo[-1] - qp[-1]
            c[s < np.min(qp)] = s[s < np.min(qp)] + qo[0] - qp[0]

    # Delta method: Apply the mean delta to the projection data
    elif method == 'delta':
        c = s + (np.nanmean(p) - np.nanmean(obs))

    # Scaling additive method
    elif method == 'scaling_add':
        c = abs(s - (np.nanmean(p) + np.nanmean(obs)))

    # Scaling multiplicative method
    elif method == 'scaling_multi':
        c = (s / np.nanmean(p)) * np.nanmean(obs)

    # BCSD method: Adjust using pre-calculated monthly factors
    elif method == 'BCSD':
        s.index = pd.DatetimeIndex(s.index)  # Ensure proper datetime index
        df_ajustado = s.copy()
        for mes in range(1, 13):  # Apply correction for each month
            df_ajustado[df_ajustado.index.month == mes] *= factor_Bcsd.loc[mes].values[0]
        c = df_ajustado

    else:
        raise ValueError("Incorrect method, choose from 'delta', 'scaling_add', 'scaling_multi', or 'eqm'")

    return c


def proceso_BCSD(df_obs, p):
    """
    Calculate the BCSD adjustment factor between observed historical data and model data.

    Parameters:
    - df_obs: Historical observed data.
    - p: Historical model data (GCM).

    Returns:
    - ajuste_factor_2: Monthly adjustment factor for BCSD correction.
    """

    df_obs.index = pd.DatetimeIndex(df_obs.index)  # Ensure datetime index for observations
    p.index = pd.DatetimeIndex(p.index)  # Ensure datetime index for model data

    # Calculate monthly means for both observed and model data
    monthly_averages_obs = df_obs.groupby(df_obs.index.month).mean()
    monthly_averages_P = p.groupby(p.index.month).mean()

    # Ensure matching column types between dataframes
    if type(monthly_averages_obs.columns[0]) != type(monthly_averages_P.columns[0]):
        monthly_averages_obs.columns = monthly_averages_obs.columns.astype(str)
        monthly_averages_P.columns = monthly_averages_P.columns.astype(str)

    # Calculate adjustment factor
    ajuste_factor_2 = monthly_averages_obs / monthly_averages_P
    return ajuste_factor_2


def bias_correction_for_station_hist(sta, file_obs, df_model_train, met='BCSD'):
    """
    Perform bias correction on historical data for a specific station.

    Parameters:
    - sta: Station name.
    - file_obs: Observed data file.
    - df_model_train: Model historical data (GCM).
    - met: Bias correction method (default: 'BCSD').

    Returns:
    - sta: Station name.
    - adjust_historical: Bias-corrected historical series.
    """

    start_period_T = file_obs[sta].dropna().index.min()  # Start period of observed data
    p_train = df_model_train[str(sta)].loc[str(start_period_T):].interpolate()  # Model historical series

    end_period = p_train.index.max()  # End period of the series
    obs_d = file_obs[sta].loc[start_period_T:end_period].interpolate()  # Observed historical series

    # Calculate BCSD adjustment factor
    Factor = proceso_BCSD(obs_d, p_train)

    # Perform bias correction
    adjust_historical = bias_correction(obs_d, p=p_train, s=df_model_train[str(sta)].loc['1970-01-01':'2014-12-31'],
                                        method=met, factor_Bcsd=Factor, nbins=1000, extrapolate='constant')

    return sta, adjust_historical


def bias_correction_for_station(sta, file_obs, df_model_train, df_model_test, met='BCSD', f_inicial='01/01/2015'):
    """
    Perform bias correction on projected data for a specific station.

    Parameters:
    - sta: Station name.
    - file_obs: Observed data file.
    - df_model_train: Model historical data (GCM).
    - df_model_test: Model projected data (GCM).
    - met: Bias correction method (default: 'BCSD').
    - f_inicial: Start date for the test period (default: '01/01/2015').

    Returns:
    - sta: Station name.
    - Adjust: Bias-corrected projected series.
    """

    start_period_T = file_obs[sta].dropna().index.min()  # Start period of observed data
    p_train = df_model_train[str(sta)].loc[str(start_period_T):].interpolate()  # Model historical series
    s_test = df_model_test[str(sta)].loc[f_inicial:].interpolate()  # Projected series for the future

    # Observed historical series
    obs_d = file_obs[sta].loc[start_period_T:p_train.index.max()].interpolate()

    # Calculate BCSD adjustment factor
    Factor = proceso_BCSD(obs_d, p_train)

    # Perform bias correction
    Adjust = bias_correction(obs_d, p=p_train, s=s_test, method=met, factor_Bcsd=Factor, nbins=1000,
                             extrapolate='constant')

    return sta, Adjust


# Paths to input and output data
BASE_PATH = r'D:\Dropbox\Trabajos\Actividades_M\PORH\07_Hidrologia_CC/'
path_cat = os.path.join(BASE_PATH, '01_Caracterizacion_Climatica_Historica')
path_series_gcm = os.path.join(BASE_PATH, '02_Escenarios_Cambio_Climatico', '01_Series_GCM_raw')
path_out = os.path.join(BASE_PATH, '02_Escenarios_Cambio_Climatico', '02_Series_Downscaling')
Name_catalogo = '03_Catalogo_Cumplen.xlsx'
Name_serie_obs = '02_Datos_Seleccionados.xlsx'

# Dictionaries for variable and catalog mappings
dict_var = {'Tas': 'tas', 'PT': 'pr'}
dict_cat = {'Tas': 'TS_1', 'PT': 'PT_4'}
escenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # Future climate scenarios
variables = ['PT', 'Tas']
meth = 'BCSD'  # Bias correction method

if __name__ == '__main__':
    for var in variables:
        print(f"Processing variable: {var}")

        # Create output directory if it doesn't exist
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        subdir = path_series_gcm.replace("\\", "/")
        models = next(os.walk(subdir))[1]  # Get model directories

        # Load station catalog
        Catalogo = pd.ExcelFile(os.path.join(path_cat, Name_catalogo)).parse(dict_cat[var], index_col=0)
        estaciones = Catalogo.index

        try:
            file_obs = \
            pd.ExcelFile(os.path.join(path_cat, Name_serie_obs)).parse(dict_cat[var], index_col=0)[
                estaciones.astype(str)]
        except:
            file_obs = \
            pd.ExcelFile(os.path.join(path_cat, Name_serie_obs)).parse(dict_cat[var], index_col=0)[
                estaciones]

        # Save historical observed data
        name_historical = os.path.join(path_out, f'01_Series_historical-obs_{dict_cat[var]}.csv')
        file_obs.to_csv(name_historical, index=True)

        for m in models:
            print(f"Processing model: {m}")
            model_path = os.path.join(path_out, m, dict_cat[var])
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            new_path = os.path.join(subdir, m, dict_cat[var])
            df_model_train = pd.read_csv(os.path.join(new_path, f'Series_historical_{dict_cat[var]}.csv'), index_col=0)

            # Perform bias correction for historical period using multiprocessing
            with Pool() as p:
                func = partial(bias_correction_for_station_hist, file_obs=file_obs, df_model_train=df_model_train,
                               met=meth)
                results = p.map(func, estaciones)

            # Save historical bias-corrected data
            results_dict = {sta: series for sta, series in results}
            df_historical = pd.DataFrame(results_dict)
            name_h_gcm = os.path.join(model_path, f'02_Ds_Series_historical-gcm_{dict_cat[var]}.csv')
            df_historical.to_csv(name_h_gcm, index=True)

            # Perform bias correction for future climate scenarios
            for i, es in enumerate(escenarios, start=3):
                print(f"Processing scenario: {es}")
                df_model_test = pd.read_csv(os.path.join(new_path, f'Series_{es}_{dict_cat[var]}.csv'), index_col=0)

                # Perform bias correction using multiprocessing
                with Pool() as p:
                    func = partial(bias_correction_for_station, file_obs=file_obs, df_model_train=df_model_train,
                                   df_model_test=df_model_test, met=meth)
                    results_gcm = p.map(func, estaciones)

                # Save bias-corrected data for the scenario
                results_dict_gcm = {sta: series for sta, series in results_gcm}
                df_gmc = pd.DataFrame(results_dict_gcm)
                name_out = os.path.join(model_path, f'0{i}_Ds_Series_{es}_{dict_cat[var]}.csv')
                df_gmc.to_csv(name_out, index=True)
