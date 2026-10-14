import numpy as np
import parse_arrays as pa
import mod_helium4_phys6152 as mod_he
import matplotlib.pyplot as plt

def convert_pv_to_pa(pressure_volts):
    '''
    Takes 1D Numpy array and converts from Volts -> Milimeters mercury -> Pascals
    Returns: 1D Numpy array of pressure in Pascals
    
    7.5V= 100,000Pa
    1V=100mmHg=13332.2 Pa
    '''
    tmp_pressure_pa = []
    for value in pressure_volts:
        value_pa = value * 13332.2
        tmp_pressure_pa.append(value_pa)

    #add tmp list to np array 
    return tmp_pressure_pa




#plan 
'''Add v_x and v_y, then average
uncertainty = vx-vy,abs value, /2 

convert dvm to temp 
temp on x, v_avg on y
'''

    
    
def known_normal_fraction_curve(temperatures):
    total_density = mod_he.density(temperatures)
    normal_fluid_density = mod_he.density_normalfluid(temperatures)
    
    density_fraction=normal_fluid_density/total_density
    return density_fraction


def plot_values():
    frequency_values,frequency_uncertainty,dvm = pa.load_file()
    
    #convert dvm voltage values to temperature
    dvm_temp = mod_he.temperature_from_pressure_SVP(convert_pv_to_pa(dvm))

    plt.figure(1)
    plt.plot(dvm_temp,frequency_values)
    plt.errorbar(dvm_temp,frequency_values,xerr=0.005,yerr=frequency_uncertainty)

    plt.title("Temperature VS Frequency (Complete dataset)")
    plt.xlabel("Temperature/K")
    plt.ylabel("frequency/Hz")
    
    min_freq_idx = frequency_values.argmin()
    
    
    print(f"minimum freq: {frequency_values[min_freq_idx]} at T = {dvm_temp[min_freq_idx]}")
    


    plt.figure(2)
    f_0 = 658 #Hz
    f_n = 453.16 #Hz find from graph
    normal_frac = ((f_0**2)/(frequency_values**2)-1)/((f_0**2)/(f_n**2)-1)
    
    peak_normal_frac_index = np.argmax(normal_frac)
    
    #cut data off here so only plot below lambda point (remove all previous values)
    cut_normal_frac = np.delete(normal_frac,range(0,peak_normal_frac_index))
    cut_dvm_temp = np.delete(dvm_temp,range(0,peak_normal_frac_index))
    
    known_density_values = known_normal_fraction_curve(cut_dvm_temp)
    plt.plot(cut_dvm_temp,cut_normal_frac,label="Experimental values")
    
    #plot known values
    plt.plot(cut_dvm_temp,known_density_values,label="Known values")
    
    plt.title("Normal Fraction density of He II VS Temperature (Complete dataset)")
    plt.xlabel("Temperature/K")
    plt.ylabel("normal_fraction")
    plt.legend()
    plt.show()


    
if __name__ == "__main__":
    plot_values()