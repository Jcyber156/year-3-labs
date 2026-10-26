import numpy as np
#L = 20800 #J/kg
t_h = 300
t_c = 4.2
surface_area = np.pi*(0.005)**2
L=1 
def equation(k):
    q = (k*surface_area*(t_h-t_c))/L
    return q 

materials = ["Copper","Constantin","Stainless steel","Nylon"]
