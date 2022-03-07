import numpy as np 
from scipy.optimize import curve_fit
class Iris():
    def __init__(self,):
        """
        Everything is in SI units
        w0 is the radius!!
        """
        self.iris_positions  = []
        self.powers = []
        self.P_max = 1
        self.w0_init = 1 #focal spot measurement at P_max
        self.f = 1
        self.wl = 1
        self.M2 = 1
        self.reprate = 1 
        self.pulse_duration = 1
        self.w_lens = 1
    
    def specify_calib(self, iris_positions, powers):
        self.iris_positions  = iris_positions
        self.powers = powers
        self.P_max = max(self.powers)

    def specify_params(self, w0_init, f,wl,M2, reprate,pulse_duration):
        self.w0_init = w0_init #focal spot measurement at P_max
        self.f = f
        self.wl = wl
        self.M2 = M2
        self.reprate = reprate 
        self.pulse_duration = pulse_duration
        self.w_lens = self.w_full(self.w0_init)

    def w_full(self,w0):
        #defines pre lens beam size
        return 4*self.M2*self.wl*self.f/(np.pi*w0)

    def w_initial(self, P_now):
        #defines irised beam size at lens for specific power  
        w2 = np.sqrt(self.w_lens**2*P_now/self.P_max)
        return w2

    def new_focus(self, P_now):
        #finds new focus spot for specific power 
        w2 = self.w_initial(P_now)
        w0_new = 4*self.M2*self.wl*self.f/(np.pi*w2)
        return w0_new

    def peak_intensity(self, P,w0):
        """
        w0: radius of the beam!!! [m]
        """
        return (2*0.94/(np.pi*w0**2))*(P/(self.reprate*self.pulse_duration))

    def sigmoid(self, x,L,x0,k,b):
        #approximate function of the iris
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return y

    def calculate_average_power(self,x):
        p0 = [max(self.powers), np.median(self.iris_positions),1,min(self.powers)]
        popt, pcov = curve_fit(self.sigmoid, self.iris_positions, self.powers, p0)
        return self.sigmoid(x, *popt)

    def get_intensity_TWcm2(self,iris_pos):
        power_new = self.calculate_average_power(iris_pos)
        w0_new = self.new_focus(power_new)
        I = self.peak_intensity(power_new, w0_new)*1e-16
        return I





