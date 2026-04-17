import config
from config import *
from Domain import *
from interface import Interface
import numpy as np
import math as math
import tensorflow as tf
import time
import os

from tensorflow import keras
from keras import layers
from keras import models
from keras import callbacks


class PINN_Model(tf.keras.Model):
    ######################################################
    ##    　　   ネットワーク層・パラメータを定義　　　　   ##
    ######################################################     
    def __init__(self, numHiddenLayers, numNeurons, domain):
        super(PINN_Model, self).__init__(name="PINN_Model")

        self.numHiddenLayers = numHiddenLayers
        self.numNeurons      = numNeurons
        self.domain = domain

        self.interface = Interface(coeff = 1e-5)
        
        self.act_coeff = tf.keras.Variable(1.0, trainable=True, dtype=config.real(tf))
   
        
        # ----------- 物性値 (pre_run_main から受取る) -----------
        self.Ga = tf.constant(93.5, dtype=config.real(tf))
#        self.We = tf.constant(4.8e-10, dtype=config.real(tf))
        self.We = tf.constant(4.8e-3, dtype=config.real(tf))
        
        self.lower_bounds = tf.constant([domain.xmin, domain.ymin, domain.tmin], dtype=config.real(tf))
        self.upper_bounds = tf.constant([domain.xmax, domain.ymax, domain.tmax], dtype=config.real(tf))

        # Define NN layers
        self.hiddenLayers = []
        
        for l in range(0, self.numHiddenLayers):
            self.hiddenLayers.append(tf.keras.layers.Dense(self.numNeurons, activation='tanh', name='Dense%s'%(l), dtype=config.real(tf)))
        self.outputLayer = tf.keras.layers.Dense(3, activation=None, name='Output', dtype=config.real(tf))
    
    #####################################
    ##    　　　NNの層の形状を構築　　  　##
    #####################################     
    # def build(self):
    #     self.hiddenLayers[0].build(input_shape=(None, 3))
    #     for l in range(1, self.numHiddenLayers):
    #         self.hiddenLayers[l].build(input_shape=(None, self.numNeurons))
            
    #     self.outputLayer.build(input_shape=(None, self.numNeurons))

      
    def build(self):
        ins=(None, 3)
        for l in range(0, self.numHiddenLayers):
            self.hiddenLayers[l].build(input_shape=ins)
            ins = (None, self.numNeurons)
           
        self.outputLayer.build(ins)

    def scale(self, X):
        aa = tf.constant(2.0, dtype=config.real(tf))
        bb = tf.constant(1.0, dtype=config.real(tf))
        ll = self.lower_bounds
        rr = self.upper_bounds
        return aa * (X - ll) / (rr - ll) - bb

  
    # def call(self, X):
    #     X = self.scale(X)
    #     for layer in self.hiddenLayers:
    #         X = layer(X)
    #     Y = self.outputLayer(X)
    #     return Y
    
    
    def call(self, X):
        X = self.scale(X)
        for layer in self.hiddenLayers:
            Y = layer(X)
            X = tf.math.tanh(self.act_coeff * Y)
        Y = self.outputLayer(X)
        return Y

   #
    def net_field(self, x, y, t):
        X = tf.concat([x,y,t], axis=1) 
        Y = self.call(X)
        u = Y[:,0:1]
        v = Y[:,1:2]
        p = Y[:,2:3]
        return u, v, p

    def Equations(self, x,y,t):
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
        
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                u, v, p = self.net_field(x, y, t)
        
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
        
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
        
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)

            # T_x = tape2.gradient(T, x)
            # T_y = tape2.gradient(T, y)

            
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1.gradient(u_y, y)
        
        v_xx = tape1.gradient(v_x, x)
        v_yy = tape1.gradient(v_y, y)

        # T_xx = tape1.gradient(T_x, x)
        # T_yy = tape1.gradient(T_y, y)

        
    
        del tape1, tape2
        
        Cnt = u_x + v_y 
        Nsx = (u*u_x + v*u_y) + p_x - (u_xx + u_yy)
        Nsy = (u*v_x + v*v_y) + p_y - (v_xx + v_yy)  + self.Ga 
        
        return Cnt, Nsx, Nsy


    ##########################################
    def call_loss_GE(self, resPoints):
        
        xx   = resPoints[:,0:1]
        yy   = resPoints[:,1:2]
        tt  = resPoints[:,2:3]

        Eqns = self.Equations(xx, yy, tt)

        # Physics loss
        loss_GE = 0.0
        for ii in range(0, len(Eqns)):
            loss_GE += tf.reduce_mean(tf.square(Eqns[ii]))
        
        return loss_GE

        
    ##############
    def call_loss_BC_Right(self, BC_Points_Right):
        x   = BC_Points_Right[:,0:1]
        y   = BC_Points_Right[:,1:2]
        t   = BC_Points_Right[:,2:3]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u, v, p = self.net_field(x, y, t)
        ux = tape.gradient(u,x)
        uy = tape.gradient(u,y)
        vx = tape.gradient(v,x)
        vy = tape.gradient(v,y)
        del tape

        nx, ny= self.interface.normal(x, y, t)
        
             # tg = - tf.math.sin(theta)
        VSMALL = 1e-13
        rr = tf.sqrt(x*x + y*y)
        r2 = tf.sqrt(x*x + y*y)
        theta = tf.math.atan2(y, x)
        
        
        p_jet = self.interface.P_jet(theta)
        tau_jet = self.interface.Tau_jet(theta)

        BC1 = u*nx + v*ny  # u \dot n = 0
        BC2 = self.interface.curvature(x,y,t) - (p + p_jet) * self.We
       
        L1 = 2.0 * ux * nx  +  (uy + vx)*ny 
        L2 = (uy + vx)* nx  +  2.0*vy*ny     
        
        t1 = -ny
        t2 =  nx
        R1 = t1
        R2 = t2
        
   
     
        BC31 = L1 - tau_jet*R1
        BC32 = L2 - tau_jet*R2

        BC1 = tf.reduce_mean(tf.square(BC1))
        BC2 = tf.reduce_mean(tf.square(BC2))
        BC3 = tf.reduce_mean(tf.square(BC31)) \
            + tf.reduce_mean(tf.square(BC32)) \
        
        BC = BC1 + BC2 + BC3
    
        return BC, BC1, BC2, BC3
    
    
    # ##########################################
    def cal_loss_pRef(self):
        x0 = np.full((1,1), 1.0, dtype=config.real(np))
        y0 = np.full((1,1), 0.0, dtype=config.real(np))
        t0 = np.full((1,1), 0.0, dtype=config.real(np))
        
        
        x0 = tf.convert_to_tensor(x0, dtype=config.real(tf))
        y0 = tf.convert_to_tensor(y0, dtype=config.real(tf))
        t0 = tf.convert_to_tensor(t0, dtype=config.real(tf))
        
        sol = self.net_field(x0, y0, t0)
        pp = sol[2]


        loss_Pref = tf.reduce_mean(tf.square(pp))
        loss_Pref = 0
        return loss_Pref

    def loss_fn(self, dataList):
        dataGE = dataList[0]                                  # shape: (N, 3)
        # dataBC_L = dataList[1]        
        dataBC_R = dataList[1]

        loss_GE = self.call_loss_GE(dataGE)
        # BC_L, BC1_L, BC2_L, BC3_L = self.call_loss_BC_Left(dataBC_L)
        BC_R, BC1_R, BC2_R, BC3_R = self.call_loss_BC_Right(dataBC_R)

        loss_pRef = self.cal_loss_pRef() 
        # BC = BC_L + BC_R
        BC =  BC_R
        

        loss_value = loss_GE + BC + loss_pRef

        return loss_value

    def sub_loss_labels(self):        
        return ["GE", "BC", "BC1", "BC2", "BC3", "pRef"]


    ###################################
    def loss_eval(self, dataList):
        dataGE = dataList[0]                                  # shape: (N, 3)
        # dataBC_Left = dataList[1]        
        dataBC_Right = dataList[1]

        loss_GE = self.call_loss_GE(dataGE)
        # BC_L, BC1_L, BC2_L, BC3_L = self.call_loss_BC_Left(dataBC_Left)
        BC_R, BC1_R, BC2_R, BC3_R = self.call_loss_BC_Right(dataBC_Right)
        loss_pRef = self.cal_loss_pRef()
        
        # BC1 = BC1_L + BC1_R
        # BC2 = BC2_L + BC2_R
        # BC3 = BC3_L + BC3_R
        BC1 =  BC1_R
        BC2 =  BC2_R
        BC3 =  BC3_R
        
        BC = BC1 + BC2 + BC3
        
        loss_value = loss_GE + BC + loss_pRef


        return loss_value, [loss_GE, BC, BC1, BC2, BC3, loss_pRef]


##############
    # def call_loss_BC_Left(self, BC_Points_Left):
    #     x   = BC_Points_Left[:,0:1]
    #     y   = BC_Points_Left[:,1:2]

    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(x)
    #         tape.watch(y)
    #         u, v, p = self.net_field(x, y)
    #     ux = tape.gradient(u,x)
    #     uy = tape.gradient(u,y)
    #     vx = tape.gradient(v,x)
    #     vy = tape.gradient(v,y)
    #     del tape

    #     nx, ny= self.interface.normal(x, y)

    #     # tg = - tf.math.sin(theta)
    #     VSMALL = 1e-13
    #     rr = tf.sqrt(x*x + y*y)
    #     r2 = tf.sqrt(x*x + y*y)
    #     theta = tf.math.atan2(y, x)
        
        
    #     p_jet = self.interface.P_jet(theta)
    #     tau_jet = self.interface.Tau_jet(theta)
    #     tau_jet = 0
        
    #     BC1 = u*nx + v*ny  # u \dot n = 0
    #     BC2 = 1.0 - p_jet * self.We
    #     # BC2 = self.interface.curvature(x,y) - (p + p_jet) * self.We
       
    #     L1 = 2.0 * ux * nx  +  (uy + vx)*ny 
    #     L2 = (uy + vx)* nx  +  2.0*vy*ny     
        
    #     t1 = -ny
    #     t2 =  nx
    #     R1 = t1
    #     R2 = t2
        
    #     #ベクトル変更
    #     # tau_jet = -tau_jet
        


    #     BC31 = L1 - tau_jet*R1
    #     BC32 = L2 - tau_jet*R2

    #     BC1 = tf.reduce_mean(tf.square(BC1))
    #     BC2 = tf.reduce_mean(tf.square(BC2))
    #     BC3 = tf.reduce_mean(tf.square(BC31)) \
    #         + tf.reduce_mean(tf.square(BC32)) \
                
    #     # BC3 = 0.0
                
                        
    #     BC = BC1 + BC2 + BC3

    #     return BC, BC1, BC2, BC3
    