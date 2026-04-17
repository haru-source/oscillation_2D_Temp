
import config
from config import *

import numpy as np
import os
import tensorflow as tf
from Domain import *
from PINN_Model import PINN_Model
from SolverSciPy import *
import time
import argparse
import time
import argparse
import keras
import math

# import vtk
# from vtk.util import numpy_support as vns

from interface import Interface

from scipy.linalg import cholesky,LinAlgError

config.set_random_seed(1234)
config.set_default_float("float64")

def print_curvature(theta, phi):
    interf = Interface(0.0)
    vx = tf.math.sin(theta) * tf.math.cos(phi)
    vy = tf.math.sin(theta) * tf.math.sin(phi)
    vz = tf.math.cos(theta)  
    print(vx.numpy(), vy.numpy(), vz.numpy())
    print(interf.curvature(vx, vy, vz).numpy())

def Test():

    print_curvature(0.0, 0.0)
    print_curvature(np.pi, 0.0)
    print_curvature(np.pi, np.pi)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',         type=str, default = 'output_TEST')
    parser.add_argument('--epochs_A',    type=int, default=30)
    parser.add_argument('--epochs_B',    type=int, default=30)
    parser.add_argument('--method',      type=str, default='BFGS')
    parser.add_argument('--method_sub',  type=str, default='SSBroyden2') # SSBryoyden2, SSBFGS_AB
    parser.add_argument('--Nf'        ,  type=int, default=1000)  
    parser.add_argument('--Nbc'       ,  type=int, default=1000)   
    parser.add_argument('--Nt'        ,  type=int, default=100)    

    args  = parser.parse_args()
    
    out_dir = args.dir
    os.makedirs(out_dir, exist_ok=True)
    print("TensorFlow version: {}".format(tf.__version__))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    rmin = 1e-6
    domain = DomainSphere(Space_1D(rmin, 1.0), Space_1D(0, np.pi ), timeDomain=TimeDomain(0.0,2*np.pi), a2=0.1)  

    model = PINN_Model(numHiddenLayers=5, numNeurons=30, domain = domain)
    model.build()
    model.summary()

    if os.path.exists('{:}/PINN.weights.h5'.format(out_dir)):
        print("loading: PINN.weights.h5")
        model.load_weights('{:}/PINN.weights.h5'.format(out_dir))

    iter_last = 0
    if os.path.exists('{:}/loss_BFGS.dat'.format(out_dir)):
        lf = np.loadtxt('{:}/loss_BFGS.dat'.format(out_dir), delimiter='\t', skiprows=1)
        iter_last = int(lf[-1,0])
        print('last iter in previous run:', iter_last)

    Nf = args.Nf
    Nbc = args.Nbc
    Nt = args.Nt
    
    
    GE_points = domain.genResidualPoints(Nf,Nt,out_dir)
    BC_points = domain.genBoundaryPoints(Nbc,Nt,out_dir)
    
    
    dataList  = [GE_points, BC_points]    
    solver = SolverSciPy(model, out_dir=out_dir, iter_ini=iter_last)

    start = time.time()


    print('#'*10, "Adam", '#'*10 )
    opt = keras.optimizers.Adam(learning_rate=1e-2)
    solver.train_Adam(dataList, epochs=args.epochs_A, optimizer=opt, lossFileHeader='loss_Adam')

    print('#'*10, "BFGS", '#'*10 )
    
    initial_weights = np.concatenate([tf.reshape(w, [-1]).numpy() for w in model.trainable_variables]) 
    H0 = np.eye(len(initial_weights), dtype=config.real(np))
    iter_sub = min(1000, max(args.epochs_B, 0))
    
    loop = 0
    while solver.get_iter() < iter_last + args.epochs_A + args.epochs_B:
        print("iter : ", solver.get_iter())
        ret = solver.train(dataList, epochs=iter_sub, lossFileHeader='loss_BFGS', 
                               method=args.method, method_sub=args.method_sub, hess_inv0=H0)
        
        if args.method != "L-BFGS-B":
            H0 = ret.hess_inv
            H0 = (H0 + np.transpose(H0)) / 2.0
            try:
                cholesky(H0)
            except LinAlgError:
                H0 = np.eye(len(initial_weights), dtype=config.real(np))

        model.save_weights('{:}/PINN.weights.h5'.format(out_dir), 'weights')

        # write(model, out_dir)
        
        loop += 1
        if loop > 50:
            print(" BFGS loop exceeded maximum")
            break

    elapse = time.time() - start
    print('Elapsed time:', elapse)

    model.save_weights('{:}/PINN.weights.h5'.format(out_dir), 'weights')
    
    # # ################################################
    # # ##                   評価(過去)                ##
    # # ################################################
    Nr_eval = 30   # r 方向
    Nq_eval = 30   # theta 方向
    Nt_eval = 100   # 時間方向
    Grid_points = domain.genGirdPoints(Nr_eval, Nq_eval, Nt_eval,out_dir="output_Grid")
    x_eval = Grid_points[:,0:1]
    y_eval = Grid_points[:,1:2]
    t_eval = Grid_points[:,2:3]
    u, v, p = model.net_field(x_eval, y_eval, t_eval)
    arrayForOutput = np.hstack((x_eval, y_eval, t_eval, u, v, p))
    np.savetxt('{:}/new_result_{:05d}.tsv'.format(out_dir, solver.get_iter()), arrayForOutput, fmt = '%.6e', delimiter = '\t', newline = '\r\n', header='x \t y \t t \t u \t v \t w \t p')
    
    

    domain.split_tsv_by_time("output_TEST/new_result_00060.tsv" , out_dir="output_split_Grid")
    print("outputdata出力完了")