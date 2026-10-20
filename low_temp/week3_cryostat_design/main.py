import numpy as np
import matplotlib.pyplot as plt

# Take equation q = rho_e * (I/A)**2 and plot to find a reasonable value of A for rho_e.
# All calculations are done in SI units; conversions happen only when plotting.

CURRENT = 1e-6   # A (assuming about one microamp for now)
RHO_E = 48e-8     # Ohm m (copper at 77 K)

# Wire diameters: 50 to 499 um, stored in metres
d_values = np.arange(50, 500) * 1e-6


def area(d):
    """Cross-sectional area (m^2) of a wire of diameter d (m)."""
    return np.pi * (d / 2) ** 2


def q_equation(a):
    """Volumetric heat generation rate in W/m^3.
    rho [Ohm m] * (I/A)^2 [A^2/m^4] = W/m^3.
    Numerically, 1 W/m^3 = 1 uW/cm^3.
    """
    return RHO_E * (CURRENT / a) ** 2


def generate_q_values():
    return q_equation(area(d_values))


#def get_resistance():
    """rho = RA/L
    R = rho*L/A"""
    l = 1 #m
    return RHO_E*l/area(d_values)


def get_thermal_conductivity():
    l=1#meters
    thermal_conductivity = (296.8)*20*area(d_values)/l 
    return thermal_conductivity

def plot_values():
    q_values = generate_q_values()               # W/m^3 == uW/cm^3
    thermal_conductivity = get_thermal_conductivity()*10**4 #uWcm^-1k^-1
    d_um = d_values * 1e6                      # same x data for both curves

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(d_um, q_values, color="C0")
    ax1.set_xlabel("Diameter / μm")
    ax1.set_ylabel(r"Volumetric heat generation rate / μW cm$^{-3}$", color="C0")
    #ax1.set_yscale("log")
    ax2.plot(d_um,thermal_conductivity,color="C1")
    ax2.set_ylabel(r"Thermal conductivty of constantan/μW m$^{-1}$K$^{-1}$",color="C1")
    plt.title("Heat generation rate vs thermal conductivity for a varying diameter constantan wire")

    plt.show()


if __name__ == "__main__":
    plot_values()