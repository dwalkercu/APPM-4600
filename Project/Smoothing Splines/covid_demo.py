import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cubic_splines as cs
import pandas as pd
from rls_ss_demo import regularized_least_squares, find_opt_lambda_rls

def awgn(signal, snr_db):
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)

    # Calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db/10))

    # Generate Gaussian noise with zero mean and calculated variance
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    return signal + noise

def driver():
    pd.options.mode.copy_on_write = True

    ''' Clear trend that can't be done using normal RLS interpolation '''
    df_o = pd.read_csv("data_table_for_weekly_deaths__the_united_states_filtered.csv")

    df_o["Date"] = pd.to_datetime(df_o["Date"])
    df_o = df_o.sort_values("Date")

    df = df_o.dropna()

    # smoothing spline
    Neval = 1000
    N = len(df)
    x = np.zeros(N)
    for i in range(N):
        x[i] = i
    x0 = np.linspace(0, N, Neval)
    data = df["Weekly Deaths"].to_numpy()
    ss_data = cs.eval_smoothing_spline(x0, x, data, lda=1e-5)

    # RLS
    rls3 = regularized_least_squares(x, data, deg=3, lda=1)
    rls4 = regularized_least_squares(x, data, deg=4, lda=1)

    # x-axis values for plotting
    date_np = df["Date"].to_numpy()
    x_dates = pd.date_range(start=date_np[0], end=date_np[-1], periods=N)
    x0_dates = pd.date_range(start=date_np[0], end=date_np[-1], periods=Neval)

    _, ax = plt.subplots()
    plt.title("Deaths in the United States from Covid-19")
    plt.xlabel("Year")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(df["Date"], df["Weekly Deaths"], label="Original Data")
    plt.legend()

    _, ax = plt.subplots()
    plt.title("Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Smoothing Spline")
    ax.plot(x_dates, rls3, 'g-', label="Regularized Least Squares Degree 3")
    ax.plot(x_dates, rls4, 'r-', label="Regularized Least Squares Degree 4")
    plt.legend()

    plt.show()

    ''' Trends from noisy data -- can we get the same trend from adding AWGN?'''
    # 10dB SNR
    noisy_data = awgn(data, 10)
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=cs.find_opt_lambda_ss(x, noisy_data, min_lda=1e-5, max_lda=10, n=100))

    _, ax = plt.subplots()
    plt.title("Noisy (10dB SNR) Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.scatter(x_dates, noisy_data, label="Noisy Data")
    plt.legend()

    _, ax = plt.subplots()
    plt.title("10dB SNR Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # 5dB SNR
    noisy_data = awgn(data, 5)
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=cs.find_opt_lambda_ss(x, noisy_data, min_lda=1e-5, max_lda=10, n=100))

    _, ax = plt.subplots()
    plt.title("Noisy (5dB SNR) Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.scatter(x_dates, noisy_data, label="Noisy Data")
    plt.legend()

    _, ax = plt.subplots()
    plt.title("5dB SNR Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()


    # 1dB SNR -- cross-validation overfits the data
    noisy_data = awgn(data, 1)
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=cs.find_opt_lambda_ss(x, noisy_data, min_lda=1e-5, max_lda=10, n=100))

    _, ax = plt.subplots()
    plt.title("Noisy (1dB SNR) Deaths in the United States from Covid-19")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.scatter(x_dates, noisy_data, label="Noisy Data")
    plt.legend()

    _, ax = plt.subplots()
    plt.title("1dB SNR Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # manual selection of lambda instead of cross-validation for 1dB SNR
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=4.8)

    _, ax = plt.subplots()
    plt.title("Manual Selection Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # slight oversmoothing for 1dB SNR
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=10)

    _, ax = plt.subplots()
    plt.title("Slight Underfitting Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()

    # extreme oversmoothing for 1dB SNR
    ss_noisy_data = cs.eval_smoothing_spline(x0, x, noisy_data, lda=100)

    _, ax = plt.subplots()
    plt.title("Heavy Underfitting Spline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()
    

if __name__ == "__main__":
    driver()