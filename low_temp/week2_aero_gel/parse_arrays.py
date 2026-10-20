import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the tab-separated .dat file
# --------------------------------------------------
def load_file():
    filename = "frequency_sweep_2026_10_13.dat"
    
    df = pd.read_csv(filename, sep="\t")
    
    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()
    
    # Display column names and first few rows
    print("Columns found:")
    print(df.columns.tolist())
    
    print("\nFirst five rows:")
    print(df.head())
    
    
    
    vx_height = df["Vx ht (nV)"].to_numpy(dtype=float)
    vx_width = df["Vx width"].to_numpy(dtype=float)
    vx_freq = df["Vx freq"].to_numpy(dtype=float)
    
    
    dvm = df["DVM"].to_numpy(dtype=float)
    
    vy_width = df["Vy width"].to_numpy(dtype=float)
    vy_freq = df["Vy freq"].to_numpy(dtype=float)
    
    
    
    # calculate frequency differences and then get error
    
    
    frequency_uncertainty = abs(vx_freq - vy_freq)/2
    frequency_values= (vx_freq+vy_freq)/2

    return frequency_values,frequency_uncertainty,dvm