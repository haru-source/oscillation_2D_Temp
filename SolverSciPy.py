######################################
### for tensorflow 2.16
######################################

import config
from config import *

import copy
import numpy as np
import tensorflow as tf
import scipy.optimize


class SolverSciPy():
    def __init__(self, model, out_dir='output', iter_ini:int = 0):
        self.model = model
        self.out_dir = out_dir

        self.shapes = tf.shape_n(model.trainable_variables)
        self.n_tensors = len(self.shapes)

        ## make structures for stitch and partition of weights
        count = 0
        self.indices = [] # stitch indeces
        self.partitions = [] # partition indices
        for i, shape in enumerate(self.shapes):
            n = np.prod(shape)
            self.indices.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
            self.partitions.extend([i] * n)
            count += n
        self.partitions = tf.constant(self.partitions)

        self.iter = iter_ini
        self.current_loss = 0.0

        self.subLossLabels = model.sub_loss_labels()
        self.numSubLoss = len(self.subLossLabels)
        self.current_subLosses = np.zeros(self.numSubLoss, dtype=config.real(np))

        self.lossFileHeader = "loss"

    def get_iter(self):
        return self.iter

    ###############################
    def to_flat_weights(self, weights):
        return tf.dynamic_stitch(self.indices, weights)

    ###############################
    @tf.function
    def set_flat_weights(self, flat_weights):
        weights = tf.dynamic_partition(flat_weights, self.partitions, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, weights)):
            self.model.trainable_variables[i].assign(tf.reshape(tf.cast(param, config.real(tf)), shape))

    ###############################
    # executed by each replica
    @tf.function
    def get_loss_grad(self, dataList):
        with tf.GradientTape() as tape:
#            tape.watch(self.model.trainable_variables)  # Trainable variables are automatically watched.
            loss = self.model.loss_fn(dataList)

        grads = tape.gradient(loss, self.model.trainable_variables)
                    
        return loss, grads
    
    ###############################
    @tf.function
    def evaluate_losses(self, dataList):
        loss, sub_losses = self.model.loss_eval(dataList)

        return loss, sub_losses
                        
    ###############################
    def train_Adam(self, dataLists, epochs, optimizer, lossFileHeader="loss_Adam"):

        def train_step():
            loss, grads = self.get_loss_grad(dataLists)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        ### Execute ###
        self.lossFileHeader = lossFileHeader 
        self.dataLists = dataLists

        lossFileName = self.out_dir + "/" + self.lossFileHeader + ".dat"
        self.initializeLoss(lossFileName, dataLists)

        for epoch in range(epochs):
            loss = train_step() 

            self.callback()

        self.loss_file.close()

    ########################################
    def train(self, dataLists, epochs, lossFileHeader="loss", 
              method='L-BFGS-B', method_sub=None,
              hess_inv0=None):
        
        self.lossFileHeader   = lossFileHeader
        self.dataLists = dataLists

        print("method = ", method, "method_sub = ", method_sub)

        #########################
        def value_and_gradients_func(w):
            self.set_flat_weights(w)
            loss, grads = self.get_loss_grad(self.dataLists)

            grad_flat = tf.dynamic_stitch(self.indices, grads)

            return loss, grad_flat.numpy().astype('float64')

        x0 = self.to_flat_weights(self.model.trainable_variables)
 
        lossFileName = self.out_dir + "/" + self.lossFileHeader + ".dat"
        self.initializeLoss(lossFileName, self.dataLists)

        if method == 'L-BFGS-B':
            results = scipy.optimize.minimize(
                fun = value_and_gradients_func,
                x0 = x0,
                jac = True,
                method = 'L-BFGS-B',
                callback = self.callback,
                options = {
                    'maxiter' : epochs,
                    'maxfun' : epochs * 5,
                    'maxls'  : 50,
                    'maxcor' : 50,
                    'iprint'  : 0,
                    'gtol'    : 0,
                    'ftol'    : 1.0 * np.finfo(float).eps
                    },
                )
            self.set_flat_weights(results.x)

        elif method == 'BFGS':
            H0 = tf.eye(len(x0), dtype=config.real(tf))
            results = scipy.optimize.minimize(
                fun = value_and_gradients_func,
                x0 = x0,
                jac = True,
                method = 'BFGS',
                callback = self.callback,
                options = {
                    'maxiter' : epochs,
                    'disp'  :  False,
                    'gtol'    : 0,
                    'xrtol'    : 1.0 * np.finfo(float).eps,
                    'hess_inv0': hess_inv0,
                    'method_bfgs': method_sub,
                    'initial_scale': False
                    },
                )

            self.set_flat_weights(results.x)

        self.loss_file.close()

        return results

    ###############################
    def callback(self, xr=None):
        if self.iter % 10 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss))

        loss, sub_losses = self.evaluate_losses(self.dataLists)
        
        self.current_loss = loss.numpy()
        for ix, val in enumerate(sub_losses):
            self.current_subLosses[ix] = val.numpy()

        self.writeCurrentLoss()

        self.iter+=1

    ###############################
    def initializeLoss(self, lossFileName, dataList):
        self.loss_file = open(lossFileName, 'a')
        if self.iter == 0:
            self.loss_file.write('Iter \t Total')
            for label in self.subLossLabels:
                self.loss_file.write('\t')
                self.loss_file.write(label)
            self.loss_file.write('\n')

        loss, sub_losses = self.evaluate_losses(dataList)
  
        self.current_loss = loss.numpy()
        for ix, val in enumerate(sub_losses):
            self.current_subLosses[ix] = val.numpy()

    ###############################
    def writeCurrentLoss(self):
        self.loss_file.write('{}'.format(self.iter))
        self.loss_file.write('\t{:10.8e}'.format(self.current_loss))

        for sub in self.current_subLosses:
            self.loss_file.write('\t')
            self.loss_file.write('{:10.8e}'.format(sub))

        self.loss_file.write('\n')

