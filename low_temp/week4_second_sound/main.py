
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import mod_helium4_phys6152 as helium


# Settings
length = 0.07260  # Resonator length in metres
generator_frequency = False  # Set to True if using generator frequencies
warming = False #If we are warming up the dewar with the heater as well


# Load cooling data

filename = "calibration.dat"

df = pd.read_csv(filename, sep="\t")

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

#remove any rows with 0 temperature

resistance_cal = df["Ge_thermometer"].to_numpy(dtype=float)
pressure = df["pressure"].to_numpy(dtype=float)

#Create a boolean mask of rows where resistance is NOT zero
valid_mask = pressure != 0

#Filter both arrays simultaneously using the mask
resistance_cal = resistance_cal[valid_mask]
pressure = pressure[valid_mask]

correct_pressure = pressure-0.5

# Print to verify shapes match
print(f"Cleaned elements remaining: {len(resistance_cal)}")
print(f"Pressures = {pressure}")

# Convert pressure from mbar to Pa, then calculate temperature
pressure = pressure * 100
temperature_cal = helium.temperature_from_pressure_SVP(pressure)

'''
# Plot the calibration measurements
plt.figure()
plt.plot(temperature_cal, resistance_cal, "o")
plt.xlabel("Temperature (K)")
plt.ylabel("Resistance (Ohms)")
plt.title("Thermometer calibration")
plt.grid()
plt.show()
'''


# Fit a calibration curve to convert resistance into temperature
indices = np.argsort(resistance_cal)
resistance_cal = resistance_cal[indices]
temperature_cal = temperature_cal[indices]

if len(np.unique(resistance_cal)) != len(resistance_cal):
    raise ValueError("There are duplicate resistance values in the calibration data.")

calibration_curve = PchipInterpolator(
    resistance_cal,
    temperature_cal,
    extrapolate=False
)

corrected_calibration_curve = PchipInterpolator(
    resistance_cal,
    helium.temperature_from_pressure_SVP(correct_pressure*100),
    extrapolate=False
)


#Create 500 evenly spaced resistance steps between your min and max values
resistance_fit = np.linspace(resistance_cal.min(), resistance_cal.max(), 500)
corrected_resistance_fit = np.linspace(resistance_cal.min(), resistance_cal.max(), 500)
# Evaluate the interpolator to find corresponding smooth temperatures
temperature_fit = calibration_curve(resistance_fit)

corrected_temperature_fit = corrected_calibration_curve(corrected_resistance_fit)
# Plot the measurements and the calibration curve together
plt.figure(figsize=(8, 6))

# Plot the raw calibration data points
#plt.plot(temperature_cal, resistance_cal, "o", label="Calibration Data", color="tab:blue")
#plt.plot(helium.temperature_from_pressure_SVP(correct_pressure*100), resistance_cal, "o", label="corrected data", color="purple")

# Plot the continuous fit line (swapping X and Y axes to match your plot alignment)
plt.plot(temperature_fit, resistance_fit, "-", label="PCHIP Fit Curve", color="tab:orange", linewidth=2)
plt.plot(corrected_temperature_fit,corrected_resistance_fit,"-",label="PCHIP Corrected Fit Curve", color="tab:cyan")


plt.xlabel("Temperature (K)")
plt.ylabel("Resistance (Ohms)")
plt.title("Thermometer Calibration with PCHIP Fit")
plt.legend()
plt.grid(True)
plt.show()



# Load warming data
'''if warming is True:
    warming = pd.read_csv("warming.csv")

    resistance = warming["Resistance"].to_numpy()
    frequency = warming["Frequency"].to_numpy()

    # The actual resonance frequency is twice the generator frequency
    if generator_frequency:
        frequency = 2 * frequency

    # Use the calibration curve to find the temperature
    temperature = calibration_curve(resistance)

    if np.any(np.isnan(temperature)):
        raise ValueError("Some resistance values are outside the calibration range.")


    # Calculate second sound velocity
    # For the fundamental mode: c2 = 2Lf
    velocity = 2 * length * frequency

    # Get the accepted values from the helium library
    accepted_velocity = helium.sound_velocity_second(temperature)


    # Print and save the results
    results = pd.DataFrame({
        "Resistance (Ohms)": resistance,
        "Temperature (K)": temperature,
        "Frequency (Hz)": frequency,
        "Measured velocity (m/s)": velocity,
        "Accepted velocity (m/s)": accepted_velocity
    })

    print(results.to_string(index=False))
    results.to_csv("second_sound_results.csv", index=False)


    # Plot measured and accepted second sound velocities
    indices = np.argsort(temperature)

    plt.figure()
    plt.plot(temperature[indices], velocity[indices], "o-", label="Measured")
    plt.plot(temperature[indices], accepted_velocity[indices], "--", label="Accepted")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Second sound velocity (m/s)")
    plt.title("Second sound velocity against temperature")
    plt.legend()
    plt.grid()
    plt.show()


    # Calculate percentage difference
    difference = 100 * (velocity - accepted_velocity) / accepted_velocity

    plt.figure()
    plt.plot(temperature, difference, "o-")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Percentage difference (%)")
    plt.title("Difference from accepted values")
    plt.grid()
    plt.show()
'''

reverse_indices = np.argsort(temperature_cal)
t_sorted = temperature_cal[reverse_indices]
r_sorted = resistance_cal[reverse_indices]

# Create the inverse interpolator: Input Temperature -> Output Resistance
curve_reverse = PchipInterpolator(
    t_sorted,
    r_sorted,
    extrapolate=False
)


resistances = [5160,4999,4895,4802,4693,4606,4470,4370,4286,4198,4098,4000,3894,3800,3680,3590,3485,3385,3289,3196,3096,
3004,2885,2792,2693,2600,2400,2185,2050 ]
for resistance in resistances:
    print(calibration_curve(resistance))