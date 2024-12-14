'''
A demo of cubic smoothing splines on an up-to-date Covid-19 dataset.
AUTHOR: Derek Walker
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import smooth_splines as ss
import pandas as pd
from lib import awgn
from rls import regularized_least_squares, find_opt_lambda

def driver():
    pd.options.mode.copy_on_write = True

    # make dataframe
    df_o = pd.read_csv("data_table_for_weekly_deaths__the_united_states_filtered.csv")

    df_o["Date"] = pd.to_datetime(df_o["Date"])
    df_o = df_o[df_o["Date"].dt.year < 2022]
    df_o = df_o.sort_values("Date")
    df = df_o.dropna()

    Neval = 1000
    N = len(df)
    num_folds = N // 10

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

    # make canonical deg 3 RLS polynomial
    rls3 = regularized_least_squares(x, data, deg=10, lda=find_opt_lambda(x, data))

    # make trigonometric deg 2 RLS polynomial
    trig_deg = 2
    omega = 3*2*np.pi/(N-1)
    M = np.zeros((N, trig_deg+1))
    for i in range(N):
        M[i][0] = x[i]
        M[i][1] = np.cos(omega*x[i])
        M[i][2] = np.sin(omega*x[i])
    
    rls_trig = regularized_least_squares(x, data, deg=trig_deg, M=M, lda=1e-8)

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
    ax.plot(x_dates, rls3, 'g-', label="Tikhonov Degree 10")
    ax.plot(x_dates, rls_trig, 'r-', label="Tikhonov Trigonometric")
    plt.legend()

    plt.show()

    # create 10dB SNR noise
    noisy_data = awgn(data, 10)
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=ss.find_opt_lambda(x, noisy_data, num_folds, min_lda=0.01, max_lda=15))

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
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=ss.find_opt_lambda(x, noisy_data, num_folds, min_lda=1, max_lda=15))

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

    # manual selection for qualtitative comparison
    ss_noisy_data = ss.eval_smoothing_spline(x0, x, noisy_data, lda=15)

    _, ax = plt.subplots()
    plt.title("2dB SNR Spline Comparison Manual Selection")
    plt.xlabel("Date")
    plt.ylabel("Weekly Deaths")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.plot(x0_dates, ss_data, 'b-', label="Clear Smoothing Spline")
    ax.plot(x0_dates, ss_noisy_data, 'g-', label="Noisy Smoothing Spline")
    plt.legend()

    plt.show()
    

if __name__ == "__main__":
    driver()