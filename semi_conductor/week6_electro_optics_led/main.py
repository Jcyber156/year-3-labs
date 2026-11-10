import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def get_data(filename):
    df = pd.read_csv(filename, sep="\t" )
    df.columns = df.columns.str.strip()
    
    diode_voltage = df["Diode Voltage"].to_numpy(dtype=float)
    diode_current = df["Diode Current"].to_numpy(dtype=float)
    photo_diode_output = df["Sensor Output (V)"].to_numpy(dtype=float)
    
    return diode_voltage, diode_current, photo_diode_output

def average_values_and_uncertainties(file_nums, filename_prefix):
    volts_list = []
    currents_list = []
    photodiode_list = []

    for num in range(file_nums): 
        full_filename = f"{filename_prefix}{num}"
        v, i, p = get_data(full_filename)
        volts_list.append(v)
        currents_list.append(i)
        photodiode_list.append(p)

    volts = np.vstack(volts_list)
    currents = np.vstack(currents_list)
    photodiode_outputs = np.vstack(photodiode_list)

    volts_average = np.mean(volts, axis=0)
    volts_uncertainty = np.ptp(volts, axis=0) / 2

    currents_average = np.mean(currents, axis=0)
    currents_uncertainty = np.ptp(currents, axis=0) / 2

    photodiode_outputs_average = np.mean(photodiode_outputs, axis=0)
    photodiode_outputs_uncertainty = np.ptp(photodiode_outputs, axis=0) / 2

    return (
        volts_average,
        volts_uncertainty,
        currents_average,
        currents_uncertainty,
        photodiode_outputs_average,
        photodiode_outputs_uncertainty,
    )

def linear_model(x, m, c):
    return m * x + c

def plot_iv_graph(file_nums, filename):
    (volts_average, volts_uncertainty, 
    currents_average, currents_uncertainty, 
    photodiode_outputs_average, photodiode_outputs_uncertainty) = average_values_and_uncertainties(file_nums, filename)
    
    # -------------------------------------------------------------
    # Plot 1: I/V Characteristics
    # -------------------------------------------------------------
    plt.figure()
    plt.errorbar(volts_average, currents_average, yerr=currents_uncertainty, xerr=volts_uncertainty, fmt="o")
    plt.xlabel("Voltage / V")
    plt.ylabel("Current / A")
    plt.title("I/V characteristics for a red LED")
    
    volts_fit_subset = volts_average[-3:]
    current_fit_subset = currents_average[-3:]
    gradient, intercept = np.polyfit(volts_fit_subset, current_fit_subset, deg=1)
    y_fit = gradient * volts_fit_subset + intercept
    
    popt, pcov = curve_fit(linear_model, volts_fit_subset, current_fit_subset, sigma=currents_uncertainty[-3:], absolute_sigma=True)

    perr = np.sqrt(np.diag(pcov))
    m_err = perr[0]  
    c_err = perr[1]  

    plt.plot(volts_fit_subset, y_fit, color="red", linewidth=2, label=f"Linear Fit (m={gradient:.4f}±{m_err:.4f})", alpha=0.7)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------
    # Plot 2: ln(I) vs V Characteristics
    # -------------------------------------------------------------
    log_current = np.log(currents_average)
    log_current_uncertainty = currents_uncertainty / currents_average
    
    plt.figure()
    
    fit_1_x_raw = volts_average[1:10]
    fit_1_y_raw = log_current[1:10]
    fit_1_sigma_raw = log_current_uncertainty[1:10]
    
    valid_mask_1 = np.isfinite(fit_1_y_raw) & np.isfinite(fit_1_sigma_raw) & (fit_1_sigma_raw > 0)
    fit_1_x = fit_1_x_raw[valid_mask_1]
    fit_1_y = fit_1_y_raw[valid_mask_1]
    fit_1_sigma = fit_1_sigma_raw[valid_mask_1]
    
    if len(fit_1_x) >= 2:
        fit_1_grad, fit_1_intercept = np.polyfit(fit_1_x, fit_1_y, deg=1)
        fit_1 = fit_1_grad * fit_1_x + fit_1_intercept
        popt, pcov = curve_fit(linear_model, fit_1_x, fit_1_y, sigma=fit_1_sigma, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        first_m_err = perr[0]  
        c_err = perr[1]  
    else:
        fit_1_grad, fit_1_intercept, first_m_err = 0, 0, 0
        fit_1 = np.array([])
    
    fit_2_x_raw = volts_average[14:30]
    fit_2_y_raw = log_current[14:30]
    fit_2_sigma_raw = log_current_uncertainty[14:30]
    
    valid_mask_2 = np.isfinite(fit_2_y_raw) & np.isfinite(fit_2_sigma_raw) & (fit_2_sigma_raw > 0)
    fit_2_x = fit_2_x_raw[valid_mask_2]
    fit_2_y = fit_2_y_raw[valid_mask_2]
    fit_2_sigma = fit_2_sigma_raw[valid_mask_2]
    
    if len(fit_2_x) >= 2:
        fit_2_grad, fit_2_intercept = np.polyfit(fit_2_x, fit_2_y, deg=1)
        fit_2 = fit_2_grad * fit_2_x + fit_2_intercept
        popt, pcov = curve_fit(linear_model, fit_2_x, fit_2_y, sigma=fit_2_sigma, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        second_m_err = perr[0]  
        c_err_2 = perr[1]  
    else:
        fit_2_grad, fit_2_intercept, second_m_err = 0, 0, 0
        fit_2 = np.array([])
    
    plt.errorbar(volts_average, log_current, yerr=log_current_uncertainty, xerr=volts_uncertainty, fmt="o", label="Data", markersize=5, capsize=3, ecolor="black")
    
    if len(fit_1) > 0:
        plt.plot(fit_1_x, fit_1, color="red", linewidth=2, label=f"Linear Fit 1 (m={fit_1_grad:.4f}±{first_m_err:.4f}), c={fit_1_intercept:.4f}", alpha=0.7)
    if len(fit_2) > 0:
        plt.plot(fit_2_x, fit_2, color="orange", linewidth=2, label=f"Linear Fit 2 (m={fit_2_grad:.4f}±{second_m_err:.4f}), c={fit_2_intercept:.4f}", alpha=0.7)
        
    plt.xlabel("Voltage / V")
    plt.ylabel("ln(Current / A)")
    plt.title("ln(I) vs V characteristics for a red LED")
    plt.legend()
    plt.show()
    
    # -------------------------------------------------------------
    # Plot 3: log(L^1/2) vs log(I) (AXES FLIPPED)
    # -------------------------------------------------------------
    # Slicing from entry 33 onwards for sensor data (Y changes to X-axis tracking)
    log_sensor_out_sqrt = (np.log(photodiode_outputs_average[33:])) * 0.5
    log_sensor_out_unc = 0.5 * (photodiode_outputs_uncertainty[33:] / photodiode_outputs_average[33:])
    
    # Slicing from entry 33 onwards for current data (X changes to Y-axis tracking)
    sliced_log_current = log_current[33:]
    sliced_log_current_unc = log_current_uncertainty[33:]
    
    # To fit from original data point 37 onwards, slice from index 4 relative to index 33 (37 - 33 = 4)
    # Flipping parameters: X is now the sensor output, Y is now the current
    x_fit_subset = log_sensor_out_sqrt[4:]
    y_fit_subset = sliced_log_current[4:]
    sigma_fit_subset = sliced_log_current_unc[4:]  # curve_fit weights by Y-uncertainty
    
    # Linear Fit calculation with flipped axes
    gradient, intercept = np.polyfit(x_fit_subset, y_fit_subset, deg=1)
    y_fit = gradient * x_fit_subset + intercept
    
    popt, pcov = curve_fit(linear_model, x_fit_subset, y_fit_subset, sigma=sigma_fit_subset, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    m_err = perr[0]  
    c_err = perr[1] 
    
    plt.figure()
    # xerr and yerr are switched to match the flipped variables
    plt.errorbar(
        log_sensor_out_sqrt, 
        sliced_log_current, 
        yerr=sliced_log_current_unc, 
        xerr=log_sensor_out_unc, 
        fmt="o", 
        label="Sensor data", 
        markersize=5, 
        capsize=3, 
        ecolor="black"
    )
    plt.plot(x_fit_subset, y_fit, color="orange", linewidth=2, label=f"Linear Fit (m={gradient:.4f}±{m_err:.4f}), c={intercept:.4f}", alpha=0.7)
    plt.xlabel(r"ln(L$^{1/2}$)")
    plt.ylabel(r"ln(Current / A)")
    plt.title("Log root of detected Lumination vs Log current")
    plt.legend()
    plt.show()

    plt.figure()
    #plot effecieny
    y_val = photodiode_outputs_average[33:]/currents_average[33:]
    y_unc = y_val*((photodiode_outputs_uncertainty[33:]/photodiode_outputs_average[33:])**2+(currents_uncertainty[33:]/currents_average[33:])**2)**(0.5)
    
    plt.errorbar(currents_average[33:],y_val,xerr=currents_uncertainty[33:],yerr=y_unc,ecolor="black",fmt="o")
    plt.xlabel("Current (A)")
    plt.ylabel("Effeciency (L/I)")
    plt.title("Effeciency vs Current")
    plt.show()
    
    
if __name__ == "__main__":
    plot_iv_graph(4, "i_v_data_")
