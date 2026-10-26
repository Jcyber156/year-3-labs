# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:13:24 2015
Modified on Thursday 28/10/2016 by TsepelinV
library should take in arguments as float, list, numpy arrays or Pandas
@author: noblemt
"""
import numpy as np
from scipy import optimize, interpolate
#import mod_oscillator as osc

########################################################################
###     Constants
########################################################################

atomic_mass_unit = 1.660539040e-27
Boltzmann_const = 1.38064852e-23
Plank_const = 6.626176e-34
Plankbar_const = Plank_const / (2.0 * np.pi)
Avogadro_const = 6.022140857e+23

molar_mass  = 4.002602e-3 #[kg]
kappa = Plank_const /(molar_mass / Avogadro_const)

T_Lambda = 2.1768 # K
density_Tlambda = 0.1461087 # g/cm**3
density_Tzero = 0.1451397 # g/cm**3
Heat_Latent_evaporation = 2.6e3 # J/l @4.2K

# Rotons constants
mass_effective_rotons = 0.16 * molar_mass * atomic_mass_unit * 1.0e3
energy_gap_rotonsK = 8.68 #[K]
velocity_rotons = 160 #[m/s]
momentum_rotons = 1.94e10 * Plankbar_const #[kg m s^{-1}]



def density(TemperatureK):
    """For a given temperature[K] function returns the helium-4 density [kg m^3].

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
   'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
   """

    # change float into numpy array for function len() to work
    try:
        len(TemperatureK)
    except:
        TemperatureK = np.array([TemperatureK])

    rho = np.tile(np.nan, len(TemperatureK)) # array of nan

    t = TemperatureK - np.tile(T_Lambda, len(TemperatureK))

    #For tempeartures below 1.334K
    #rho = SUM(from 1 = 1 to 4)m_i * T^(i+1) + rho_0
    m1 = np.array([ 8.75342, -16.7461, 7.12413, -1.26935, 0, 0]) * 1e-5
    rho_below1p334 = np.polyval(m1, TemperatureK) + density_Tzero

    #For temperatures between 1.334K and the Lambda point
    a_belTl = np.array([6.87483, -7.57537, 0]) * 1e-3
    b_belTl = np.array([0, 0, 0, 0, 4.88345, 1.86557, 3.79937, 0]) * 1e-3
    delta_rho_belTl = (np.polyval(a_belTl,t) * np.log(np.absolute(t)) + np.polyval(b_belTl,t))

    #For temperatures above the lambda point
    a_abTl = np.array([ 5.057051, -7.94605,0]) * 1e-3
    b_abTl = np.array([-0.308182, 1.53454, -2.45749, 0.240720, -3.00636, -10.2326, -30.3511,0]) * 1e-3
    delta_rho_abTl = (np.polyval(a_abTl,t) * np.log(np.absolute(t)) + np.polyval(b_abTl,t))

    rho_above1p334 = density_Tlambda * (1 + (t<0)*delta_rho_belTl + (t>0)* delta_rho_abTl)

    rho = (TemperatureK > np.tile(1.334,len(TemperatureK))) * rho_above1p334 + (TemperatureK < np.tile(1.334,len(TemperatureK))) * rho_below1p334

    return rho * 1000


def density_superfluid(TemperatureK):
    """Performs spline fit to interpolate the superfluid density.

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    k = np.array([0.0, 0.0, 0.0 ,0.0, 0.443, 0.9012, 1.5419, 1.7540, 1.918,
                  2.111, 2.156991, 2.173218, 2.175647, 2.176358, 2.176568,
                  2.176692, 2.176766, 2.176791, 2.176798, 2.176799,
                  2.17679999, 2.1768, 2.1768, 2.1768, 2.1768])

    c = np.array([1.451275432822459E-1, 1.451334563362309E-1,
                  1.449759191497576E-1, 1.455008000684433E-1, 1.4075E-1,
                  1.095E-1, 8.15E-2, 5.30E-2, 2.1E-2, 8.904576E-3, 3.053214E-3,
                  1.494043E-3, 8.342826E-4, 5.10686E-4, 2.8379E-4, 1.287426E-4,
                  5.202569E-5, 2.153580E-5, 8.564206E-6, 3.567958E-6,0])

    # change float into numpy array for function len() to work
    try:
        len(TemperatureK)
    except:
        TemperatureK = np.array([TemperatureK])

    density_superfluid = np.tile(np.nan,len(TemperatureK)) # array of nan

    for tempindex in range(len(density_superfluid)):
        density_superfluid[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3), ext = 1) * 1e3 #kg/m**3 #type:ignore

    return density_superfluid

def density_normalfluid(TemperatureK):
    """Uses the suprefluid density to work out the normal fluid density.
   Returns the whole fluid density above the lambda point."""
    #Works out whole fluid density and subtracts the superfluid density.

    rho_he = density(TemperatureK)
    rho_sup_he = density_superfluid(TemperatureK)

    return np.subtract(rho_he, rho_sup_he)


def viscosity(TemperatureK):
    """For a given temperature[K] array returns the helium-4 viscosity[Pa s].
    not very accurate?

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """
    k = np.array([7.913364e-1, 7.913364e-1, 7.913364e-1, 7.913364e-1,
                  9.705100e-1, 1.064730, 1.285930, 1.582100, 1.747010, 2.02568,
                  2.051740, 2.146961, 2.17680, 2.17680, 2.17680, 2.212906,
                  2.221800, 2.618000, 3.253700, 3.784200, 4.025400, 4.406982,
                  4.406982, 4.406982, 4.406982])

    c =  np.array([1.730865E-5, 6.577810E-6, 4.956473E-6, 1.862435E-6,
                   1.452672E-6, 1.308345E-6, 1.273173E-6, 1.338821E-6,
                   1.613257E-6, 1.956558E-6, 2.296259E-6, 2.514817E-6,
                   2.487748E-6, 2.715638E-6, 3.125798E-6, 3.487019E-6,
                   3.564378E-6, 3.486451E-6, 3.270547E-6, 3.226615E-6,
                   3.160000E-6])

    # change float into numpy array for function len() to work
    try:
        len(TemperatureK)
    except:
        TemperatureK = np.array([TemperatureK])

    viscosity = np.tile(np.nan,len(TemperatureK)) # array of nan

    for tempindex in range(len(viscosity)):
        viscosity[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3))

    return viscosity # Pa s

def pressure_SVP(TemperatureK):
    """For a given temperature[K] array returns the helium-4 vapour pressure[Pa].

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    k = np.array([0.5, 0.5, 0.5, 0.5, 0.52, 0.53, 0.56, 0.6, 0.65, 0.7, 0.75,
                  0.8, 0.85, 0.92, 1, 1.1, 1.25, 1.5, 1.7, 1.85, 2, 2.2, 2.5,
                  2.7, 3, 3.3, 3.7, 4.05, 4.5, 5.1, 5.1, 5.1, 5.1])

    c = np.array([0.00183797, 0.0025, 0.00376, 0.00624, 0.01236, 0.03306,
                   0.09339, 0.25856, 0.6269, 1.37025, 2.93305, 6.37318,
                   15.0042, 38.057, 116.050, 362.7, 990.9, 1886, 3194, 5569.5,
                   9279.1, 15370, 23480, 37355, 57050, 87170, 132825, 179650,
                   211567])

    # change float into numpy array for function len() to work
    try:
        len(TemperatureK)
    except:
        TemperatureK = np.array([TemperatureK])

    PressurePa = np.tile(np.nan,len(TemperatureK)) # array of nan

    for tempindex in range(len(PressurePa)):
        PressurePa[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3))

    return PressurePa # Pa


def temperature_from_pressure_SVP(PressurePa):
    """For a given (helium-4 vapour) pressure [Pa] returns corresponding temperature [K].
-    will return an array of temperature values for an 1D array of pressures
-
-    based on Russell J. Donnelly and Carlo F. Barenghi
-    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
-    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
-    """

    # change float into numpy array for function len() to work
    try:
        len(PressurePa)
    except:
        PressurePa = np.array([PressurePa])

    Temperature = np.tile(np.nan,len(PressurePa)) # array of nan

    P_lambda = pressure_SVP(T_Lambda)[0] # pressure at lambda point

    A_beltl = np.array([1.392408,0.527153,0.166756,0.050988,0.026514,0.001975,
                        -0.017976,0.005409,0.013259,0.0])
    B_beltl = 5.6
    C_beltl = 2.9

    A_abtl = np.array([3.146631,1.357655,0.413923,0.091159,0.016349,0.001826,
                       -0.004325,-0.004973,0.0,0.0])
    B_abtl = 10.3
    C_abtl = 1.9

    powers = np.arange(0, 10)

    for tempindex in range(len(PressurePa)):
        t_ab = (np.log(PressurePa[tempindex]) - B_abtl) / C_abtl
        t_bl = (np.log(PressurePa[tempindex]) - B_beltl) / C_beltl

        t_a = np.power(np.tile(t_ab,len(powers)), powers)
        t_b = np.power(np.tile(t_bl,len(powers)), powers)

        T_ab = np.tensordot(t_a, A_abtl, axes = 1)
        T_bl = np.tensordot(t_b, A_beltl, axes = 1)

        Temperature[tempindex] = np.where((PressurePa[tempindex] <= P_lambda), T_bl, T_ab)

    return Temperature # [K]

def sound_velocity_first(TemperatureK):
    """
    Calculates the first sound velocity of helium-4 for a given temperature

    :param TemperatureK: The temperature in Kelvin
    :return: The first sound velocity in m/s
    """
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.5016077, 0.7021246, 1.002777, 1.804065, 2.004234, 2.154563, 2.169604, 2.1758,
                  2.1763, 2.17675, 2.176797, 2.176797, 2.176797, 2.17681, 2.17683, 2.17695, 2.1773, 2.178, 2.184625,
                  2.224629, 2.505156, 4.248035, 4.248035, 4.248035, 4.248035])
    c = np.array([2.3821, 2.382041, 2.382203, 2.382958, 2.375414, 2.36402, 2.270033, 2.221461, 2.187201, 2.18001,
                  2.176634, 2.175663, 2.174367, 2.173641, 2.175712, 2.176713, 2.178168, 2.179702, 2.181464, 2.187212,
                  2.193106, 2.216496, 2.238243, 2.056185, 1.795017]) * 1e2

    try:
        len(TemperatureK)
    except TypeError as e:
        TemperatureK = np.array([TemperatureK])

    first_sound = np.zeros(len(TemperatureK))

    for temp_index in range(len(TemperatureK)):
        first_sound[temp_index] = interpolate.splev(TemperatureK[temp_index], (k, c, 3))

    return first_sound


def sound_velocity_second(TemperatureK):
    """
    Calculates the second sound velocity of helium-4 for a given temperature

    :param TemperatureK: The temperature in Kelvin
    :return: The second sound velocity in m/s

	based on Donnelly and Barenghi
    """
    k = np.array([0.5517426, 0.5517426, 0.5517426, 0.5517426, 0.6419700, 0.8062300, 0.9297000, 1.0696400, 1.285500,
				  1.720200, 1.940430, 2.068850, 2.126220, 2.153079, 2.164207, 2.169182, 2.170551, 2.174832, 2.175738,
				  2.176319, 2.176600, 2.176735, 2.176780, 2.176787, 2.176797, 2.1768, 2.1768, 2.1768, 2.1768, 2.1768])
    c = np.array([1.050000e2, 9.287425e1, 4.973642e1, 2.712778e1, 1.931258e1, 1.795336e1, 1.923863e1, 2.115625e1,
				  1.921708e1, 1.575245e1, 1.192157e1, 8.953338, 6.771458, 5.506380, 4.456879, 3.840343, 2.542421,
				  1.957438, 1.427086, 1.001273, 6.743104e-2, 4.565972e-2, 3.519492e-2, 2.088277e-2, 1.059874e-2, 0.0])

    try:
        len(TemperatureK)
    except TypeError as e:
        TemperatureK = np.array([TemperatureK])

    second_sound = np.zeros(len(TemperatureK))

    for temp_index in range(len(TemperatureK)):
        second_sound[temp_index] = interpolate.splev(TemperatureK[temp_index], (k, c, 3))

    return second_sound

def resonance_width_cylinder(radius_m, amplitude_m, length_m, viscosity, mass_effective_kg, viscous_penetration_length):
    """
-    not finished properly 28/10/2016
-
-    Returns the theoretical value for the hydrodynamic resonance width.
-
-    Fit from Grishenko paper.
-    """

    coeff_C = 0.577 # Euler constant

    delta_f = 4 * np.pi * length_m * viscosity / (2 * np.pi * mass_effective_kg * (0.5 - coeff_C - np.log( 2*radius_m * amplitude_m / viscous_penetration_length**2)))
    return delta_f

def resonance_width_phonons(diameter_beam, density_beam, TemperatureK, GeometryConstant_beam = 2.67):
    """
    Returns the theoretical value for the phonon resonance width in Helium-4 at SVP.

    diameter_beam: beam dimension
    density_beam: average density of the beam
    TemperatureK: temperature of helium
    GeometryConstant_beam: geometrical coefficient: C = 2.67 for a cylinder

    """

    try:
        len(TemperatureK)
    except TypeError as e:
        TemperatureK = np.array([TemperatureK])

    width_phonons = np.tile(np.nan,len(TemperatureK)) # array of nan

    for temp_index in range(len(TemperatureK)):
        width_phonons[temp_index] = GeometryConstant_beam / 45.0 * Boltzmann_const**4 / \
                               (Plankbar_const**3 * sound_velocity_first(TemperatureK[temp_index])**4 * \
                    diameter_beam *(density_beam + density_superfluid(TemperatureK[temp_index]))) * TemperatureK[temp_index]**4

    if len(TemperatureK) == 1:
        return width_phonons[0]
    else:
        return width_phonons

def resonance_width_rotons(diameter_beam, density_beam, TemperatureK, GeometryConstant_beam = 2.67, momentum_rotons = momentum_rotons,
                           velocity_rotons = velocity_rotons, mass_effective_rotons = mass_effective_rotons, energy_gap_rotonsK = energy_gap_rotonsK):
    """
    Returns the theoretical value for the roton resonance width in Helium-4 at SVP.

    diameter_beam: beam dimension
    density_beam: average density of the beam
    TemperatureK: temperature of helium
    GeometryConstant_beam: geometrical coefficient: C = 2.67 for a cylinder
    #Edited by Andrew Guthrie 15/2/18 - to be checked

    """

    try:
        len(TemperatureK)
    except TypeError as e:
        TemperatureK = np.array([TemperatureK])

    width_rotons = np.tile(np.nan,len(TemperatureK)) # array of nan

    for temp_index in range(len(TemperatureK)):
        width_rotons[temp_index] = (((GeometryConstant_beam * velocity_rotons * momentum_rotons**4 / Plankbar_const**4 * Plankbar_const**1) /
        (3*np.pi** 2 * diameter_beam * (density_beam + density_superfluid(TemperatureK[temp_index])))) *
        np.sqrt((mass_effective_rotons) / (8 * np.pi ** 3 * Boltzmann_const * TemperatureK[temp_index])) * np.exp(-energy_gap_rotonsK/TemperatureK[temp_index]))


    if len(TemperatureK) == 1:
        return width_rotons[0]
    else:
        return width_rotons

def resonance_width_acoustic_dipole(diameter_beam, density_beam, frequency_beam, TemperatureK, GeometryConstant = 1):
    """
    Returns the theoretical value for the acoustic resonance width in Helium-4 at SVP.

    diameter_beam: beam dimension
    density_beam: average density of the beam
    TemperatureK: temperature of helium
    GeometryConstant_beam: geometrical coefficient: C = 2.67 for a cylinder
    #Edited by Andrew Guthrie 15/2/18 - to be checked

    """

    try:
        len(TemperatureK)
    except TypeError as e:
        TemperatureK = np.array([TemperatureK])

    width_acoustic = np.tile(np.nan,len(TemperatureK)) # array of nan

    for temp_index in range(len(width_acoustic)):
        width_acoustic[temp_index] = GeometryConstant * np.pi**3 * density(TemperatureK[temp_index]) * diameter_beam**2 * frequency_beam**3 / \
        (2 * density_beam * sound_velocity_first(TemperatureK[temp_index])**2)

    if len(width_acoustic) == 1:
        return width_acoustic[0]
    else:
        return width_acoustic

def resonance_width_acoustic_dipole_vs_frequency(diameter_beam, density_beam, frequency_beam, TemperatureK, GeometryConstant = 1):
    """
    Returns the theoretical value for the acoustic resonance width as a function of frequency in Helium-4 at SVP.

    diameter_beam: beam dimension
    density_beam: average density of the beam
    TemperatureK: temperature of helium
    GeometryConstant_beam: geometrical coefficient: C = 2.67 for a cylinder
    #Edited by Andrew Guthrie 15/2/18 - to be checked

    """

    try:
        len(frequency_beam)
    except TypeError as e:
        frequency_beam = np.array([frequency_beam])

    width_acoustic = np.tile(np.nan,len(frequency_beam)) # array of nan

    for temp_index in range(len(width_acoustic)):
        width_acoustic[temp_index] = GeometryConstant * np.pi**3 * density(TemperatureK) * diameter_beam**2 * frequency_beam[temp_index]**3 /\
                                                        (2 * density_beam * sound_velocity_first(TemperatureK)**2)

    if len(width_acoustic) == 1:
        return width_acoustic[0]
    else:
        return width_acoustic

p=[100,2000]

print(temperature_from_pressure_SVP(p))

"""
def vel_error(qerr, q, amperr, amp, widtherr, width, v, merr = 0.02):

    verr = np.multiply(np.divide(v, 2),
                       np.sqrt(np.add(np.add(np.power(np.divide(qerr,q),2),np.power(np.divide(amperr,amp),2)),np.add(np.power(np.divide(widtherr,width),2),merr**2))))

    return verr
"""
