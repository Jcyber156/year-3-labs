import pandas as pd

# --------------------------------------------------
# 1. Load the tab-separated .dat file
# --------------------------------------------------
def load_file(filename):
    df = pd.read_csv(filename, sep="\t")
    
    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()
    
    vx_height = df["Vx ht (nV)"].to_numpy(dtype=float)
    vx_width = df["Vx width"].to_numpy(dtype=float)
    vx_freq = df["Vx freq"].to_numpy(dtype=float)
    
    
    dvm = df["DVM"].to_numpy(dtype=float)
    
    vy_width = df["Vy width"].to_numpy(dtype=float)
    vy_freq = df["Vy freq"].to_numpy(dtype=float)
    
    
    
    # calculate frequency differences and then get error
    
    
    frequency_uncertainty = abs(vx_freq - vy_freq)/2
    frequency_values= vx_freq
    v_width_uncertainty = abs(vx_width-vy_width)/2

    return frequency_values,frequency_uncertainty,dvm,vx_width,v_width_uncertainty

def load_drive_file(drive_up):
    dup = pd.read_csv(drive_up, sep="\t") 
        
    # Remove accidental spaces from column names
    dup.columns = dup.columns.str.strip()
    
    
    drive_up_amp = dup["ampl up mV"].to_numpy(dtype=float) #plot on x
    drive_up_vx = dup["Vx up uV"].to_numpy(dtype=float) #plot on y
    
    #load down
    drive_down_amp = dup["ampl down mV"].to_numpy(dtype=float)
    drive_down_vx = dup["Vx down uV"].to_numpy(dtype=float)
    
    return drive_up_amp*10**-3, drive_up_vx*10**-6, drive_down_amp*10**-3,drive_down_vx*10**-6 #convert to SF
    