
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import mod_helium4_phys6152 as helium


# Settings
length = 0.070  # Resonator length in metres
generator_frequency = False  # Set to True if using generator frequencies


# Load cooling data
calibration = pd.read_csv("calibration.csv")

resistance_cal = calibration["Resistance"].to_numpy()
pressure = calibration["Pressure"].to_numpy()

# Convert pressure from mbar to Pa, then calculate temperature
pressure = pressure * 100
temperature_cal = helium.temperature_from_pressure_SVP(pressure)


# Plot the calibration measurements
plt.figure()
plt.plot(temperature_cal, resistance_cal, "o")
plt.xlabel("Temperature (K)")
plt.ylabel("Resistance (Ohms)")
plt.title("Thermometer calibration")
plt.grid()
plt.show()


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


# Load warming data
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
