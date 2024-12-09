'''
A demo of cubic smoothing splines on an up-to-date Covid-19 dataset.
AUTHOR: Derek Walker
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import smooth_splines as ss
import pandas as pd
from rls import regularized_least_squares, find_opt_lambda

def awgn(signal, snr_db):
    """Returns the noisy signal using additive white Gaussian noise

    signal - the signal to add noise to
    snr_db - the signal-to-noise ratio in dB
    """
    # calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db/10))

    # generate Gaussian noise with zero mean with variance sqrt(noise_power)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # make noisy signal
    return signal + noise

def driver():
    pd.options.mode.copy_on_write = True

    ''' Clear trend that can't be done using normal RLS interpolation '''
    # make dataframe
    df_o = pd.read_csv("data_table_for_weekly_deaths__the_united_states_filtered.csv")

    df_o["Date"] = pd.to_datetime(df_o["Date"])
    df_o = df_o.sort_values("Date")
    df = df_o.dropna()

    Neval = 1000
    N = len(df)

    # make smoothing spline
    x = np.zeros(N)
    for i in range(N):
        x[i] = i
    x0 = np.linspace(0, N, Neval)
    data = df["Weekly Deaths"].to_numpy()
    ss_data = ss.eval_smoothing_spline(x0, x, data, lda=1e-5)
   
    # x-axis values for plotting dates nicely
    date_np = df["Date"].to_numpy()
    x_dates = pd.date_range(start=date_np[0], end=date_np[-1], periods=N)
    x0_dates = pd.date_range(start=date_np[0], end=date_np[-1], periods=Neval)

    # make RLS polynomials
    rls3 = regularized_least_squares(x, data, deg=3, lda=find_opt_lambda(x, data))

    # make sinusoidal RLS polynomial
    rls4 = regularized_least_squares(x, data, deg=4, lda=find_opt_lambda(x, data))

    # plot original data
    _, ax = plt.subplots()
    plt.title("Weekly Deaths in the United States from Covid-19")
    plt.xlabel("Year")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(df["Date"], df["Weekly Deaths"], label="Original Data")
    plt.legend()

    # plot spline vs RLS comparison
    _, ax = plt.subplots()
    plt.title("Weekly Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Smoothing Spline")
    ax.plot(x_dates, rls3, 'g-', label="Regularized Least Squares Degree 3")
    ax.plot(x_dates, rls4, 'r-', label="Regularized Least Squares Degree 4")
    plt.legend()

    plt.show()

    ''' Trends from noisy data -- can we get the same trend from adding AWGN?'''
    # create 10dB SNR noise
    noisy_data = awgn(data, 10)
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=ss.find_opt_lambda(x, noisy_data, min_lda=1e-5, max_lda=1, n=100))

    # plot 10dB SNR spline comparison
    _, ax = plt.subplots()
    plt.title("10dB SNR Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # 2dB SNR
    noisy_data = awgn(data, 2)
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=ss.find_opt_lambda(x, noisy_data, min_lda=1, max_lda=10, n=100))

    # plot 2dB SNR noisy data
    _, ax = plt.subplots()
    plt.title("Noisy (2dB SNR) Weekly Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.scatter(x_dates, noisy_data, label="Noisy Data")
    plt.legend()

    # plot 2dB SNR spline comparison
    _, ax = plt.subplots()
    plt.title("2dB SNR Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # manual selection of lambda instead of cross-validation for 2dB SNR
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=15.0)

    # plot manual selection lambda
    _, ax = plt.subplots()
    plt.title("Manual Selection Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # extreme oversmoothing for 2dB SNR
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=300)

    # plot manual selection
    _, ax = plt.subplots()
    plt.title("Heavy Underfitting Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()
    

if __name__ == "__main__":
    driver()