import numpy as np
import parse_arrays as pa
import mod_helium4_phys6152 as mod_he
import matplotlib.pyplot as plt
import math

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

class PlotWidths:
    def __init__(self):
        frequency_values,frequency_uncertainty,dvm,vx_width,v_width_uncertainty = pa.load_file("frequency_sweep_20261103.dat")
        self.ids = np.array([10, 13, 15, 17, 19, 21, 23, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 61, 63, 66, 68, 71])
        self.pressure_errors_to_add = np.array([1, 2, 2, 2, 2, 2, 2, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.02, 0.05]+14*[0.05])*100
        
        # Get rid of un-neaded values
        self.frequency_values = self.clean_data(frequency_values)
        self.frequency_uncertainty = self.clean_data(frequency_uncertainty)
        dvm = self.clean_data(dvm)
        self.vx_width = self.clean_data(vx_width)
        self.v_width_uncertainty = self.clean_data(v_width_uncertainty)
        
        dvm_in_pa = convert_pv_to_pa(dvm)
        self.temperature_uncertainties = self.find_temp_errors(dvm_in_pa,self.pressure_errors_to_add)
        self.temperature = mod_he.temperature_from_pressure_SVP(dvm_in_pa)


    def plot_frequency_temp(self):
        plt.errorbar(self.temperature,self.frequency_values,yerr=self.frequency_uncertainty,xerr=self.temperature_uncertainties)
        plt.title("Temperature vs Frequency for resonating NbTi Wire")
        plt.xlabel("Temperature / K")
        plt.ylabel("Frequency / Hz")
        plt.show()

    def find_temp_errors(self,dvm_pa,pressures_to_add):
        final_temps = []
        for i,pressure in enumerate(dvm_pa):
            t_plus = mod_he.temperature_from_pressure_SVP(pressure+pressures_to_add[i])
            t_minus = mod_he.temperature_from_pressure_SVP(pressure-pressures_to_add[i])
            temp_unc = abs(t_plus-t_minus)/2
            final_temps.append(temp_unc.item())
        return final_temps

    def plot_width_vs_temperature(self):
        plt.errorbar(self.temperature,self.vx_width,xerr=self.temperature_uncertainties,yerr=self.v_width_uncertainty) 
        plt.title("Temperature vs Resonant Width for NbTi Wire")
        plt.xlabel("Temperature / K")
        plt.ylabel("Resonant Width / Hz ")
        plt.show()


    def clean_data(self,data):
        #Only want data from set indicies to be included where we kept the cryostat around the same temperature
        cleaned_data = []
        for i in self.ids:
            cleaned_data.append(data[i])
        return cleaned_data



class PlotDriveData:
    def __init__(self) -> None:
        self.drive_up_amp, self.drive_up_vx, self.drive_down_amp,self.drive_down_vx = pa.load_drive_file("Drive3.dat")
        self.B = 100e-3 #magnetic field
        self.D = 3e-3 #wire loop diameter
        
        #Calulated from logbook
        self.v_x_uncertainty_frac = math.sqrt(3)/100
        self.drive_amp_uncertainty_frac = math.sqrt(3)/100


    def get_drive_force(self,drive_amp):
        frac = (np.pi*self.B*self.D)/(4*6600) #6600 comes from wire resistances
        return frac * drive_amp


    def get_velocity(self,vx): #vx is the faraday voltage
        pi_4 = np.pi/4
        return vx/(pi_4*self.B*self.D)
    
    def plot_drive_up_only_vs_velocity(self):
        drive_up_force = self.get_drive_force(self.drive_up_amp)*10**6 #change to micro newtons
        velocity = self.get_velocity(self.drive_up_vx)
        
        drive_unc = drive_up_force*self.drive_amp_uncertainty_frac
        velocity_unc = velocity*self.v_x_uncertainty_frac

        plt.errorbar(drive_up_force,velocity,xerr=drive_unc,yerr=velocity_unc,ecolor="black",elinewidth=1,capsize=1)
        plt.title("Drive up force vs velocity")
        plt.xlabel(r"$F{_d}$ / $\mu$N")
        plt.ylabel(r"Wire velocity $ms^{-1}$")
        plt.show()
        
    def plot_drive_up_with_drive_down(self):
        drive_up_force = self.get_drive_force(self.drive_up_amp)*10**6 #change to micro newtons
        drive_down_force = self.get_drive_force(self.drive_down_amp)*10**6
        velocity_up = self.get_velocity(self.drive_up_vx)
        velocity_down = self.get_velocity(self.drive_down_vx)
        
        #errors
        drive_up_f_error = drive_up_force*self.drive_amp_uncertainty_frac
        drive_down_f_error = drive_down_force*self.drive_amp_uncertainty_frac
        
        velocity_up_error = velocity_up*self.v_x_uncertainty_frac
        velocity_down_error = velocity_down*self.v_x_uncertainty_frac
        
        plt.errorbar(drive_up_force,velocity_up,xerr=drive_up_f_error,yerr=velocity_up_error,label="Drive Up data",color="blue",linestyle="-",ecolor="orange",elinewidth=1,capsize=1)
        plt.errorbar(drive_down_force,velocity_down,xerr=drive_down_f_error,yerr=velocity_down_error,label="Drive Down data",color="red",linestyle="-",ecolor="black",elinewidth=1,capsize=1)
        
        plt.title("Drive force vs wire velocity")
        plt.xlabel(r"$F{_d}$ / $\mu$N")
        plt.ylabel(r"Wire velocity $ms^{-1}$")
        
        plt.legend()
        plt.show()




if __name__ == "__main__":
    '''plot = PlotWidths()
    plot.plot_frequency_temp()
    plot.plot_width_vs_temperature()'''
    plotdrive = PlotDriveData()
    plotdrive.plot_drive_up_only_vs_velocity()
    plotdrive.plot_drive_up_with_drive_down()
