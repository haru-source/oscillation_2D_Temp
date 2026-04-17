import config
from config import *

from Domain import *

import numpy as np
import math as math
import tensorflow as tf


class Interface():
    def __init__(self, coeff):

        self.coeff = coeff
        self.coeff_Pj = 0.1
        self.coeff_Tj = 0.1

    ##########################
    def S(self, theta):
        val = 3.0 * tf.cos(theta) ** 2 - 1.0

        return self.coeff * val
    
    def F(self, x, y, t):
        eps= 1e-13
        rr = tf.sqrt(x*x + y*y + eps)
        r = tf.sqrt(rr)
        
        theta = tf.math.acos(y / (rr + eps))
        
        P2 = 3.0 * (y**2 / r) - 1.0
        a2 = 0.4
        Rmax_theta = 1.0 + a2 * P2 * tf.cos(t)
        
        F = rr - Rmax_theta

        return F
    #########################
    
    def P_jet(self, theta):
        p_jet = self.coeff_Pj * tf.math.cos(np.pi - theta)
        p_jet=0
        return p_jet
    
    ##################################
    def Tau_jet(self, theta):
        tau_jet = -self.coeff_Tj * tf.math.sin(theta)
        tau_jet=0
        return tau_jet

    
    ##########################
    def normal(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            F = self.F(x, y, t)
            
        Fx = tape.gradient(F, x)
        Fy = tape.gradient(F, y)

        denom = tf.math.sqrt(Fx**2 + Fy**2)
        Fx /= denom
        Fy /= denom
        
        del tape
        
        return Fx, Fy
    
    ##########################
    def curvature(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            nx, ny = self.normal(x, y, t)
            
        nxx = tape.gradient(nx, x)
        nyy = tape.gradient(ny, y)
        
        del tape
    
        kappa = nxx + nyy

        return kappa
    
    

    
    

