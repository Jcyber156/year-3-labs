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

class Plot:
    def __init__(self):
        frequency_values,frequency_uncertainty,dvm,vx_width,v_width_uncertainty = pa.load_file("frequency_sweep_20261103.dat")
        self.ids = np.array([10, 13, 15, 17, 19, 21, 23, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 61, 63, 66, 68, 71])
        self.pressure_errors = np.array([1, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.02, 0.05]+14*[0.05])
        
        # Get rid of un-neaded values
        self.frequency_values = self.clean_data(frequency_values)
        self.frequency_uncertainty = self.clean_data(frequency_uncertainty)
        self.dvm = self.clean_data(dvm)
        self.vx_width = self.clean_data(vx_width)
        self.v_width_uncertainty = self.clean_data(v_width_uncertainty)
        
        
        self.temperature = mod_he.temperature_from_pressure_SVP(convert_pv_to_pa(self.dvm))
        self.temperature_uncertainties = mod_he.temperature_from_pressure_SVP(self.pressure_errors)

    def plot_frequency_temp(self):
        plt.errorbar(self.temperature,self.frequency_values,yerr=self.frequency_uncertainty)
        plt.title("Temperature vs Frequency for resonating NbTi Wire")
        plt.xlabel("Temperature / K")
        plt.ylabel("Frequency / Hz")
        plt.show()


    def plot_width_vs_temperature(self):
        plt.errorbar(self.temperature,self.vx_width) #TODO investigate error odities
        plt.title("Temperature vs Resonant Width for NbTi Wire")
        plt.xlabel("Temperature / K")
        plt.ylabel("Resonant Width ")
        plt.show()


    def clean_data(self,data):
        #Only want data from set indicies to be included where we kept the cryostat around the same temperature
        cleaned_data = []
        for i in self.ids:
            cleaned_data.append(data[i])
        return cleaned_data

if __name__ == "__main__":
    plot = Plot()
    plot.plot_frequency_temp()
    plot.plot_width_vs_temperature()
