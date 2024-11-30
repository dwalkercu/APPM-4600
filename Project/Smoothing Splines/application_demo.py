import numpy as np
import matplotlib.pyplot as plt
import cubic_splines as cs
import pandas as pd
from rls_ss_demo import regularized_least_squares

def remove_outliers(df, column, threshold=1.5):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def driver():
    pd.options.mode.copy_on_write = True
    df_o = pd.read_csv("MPEA_dataset_filtered_final.csv")
    


if __name__ == "__main__":
    driver()