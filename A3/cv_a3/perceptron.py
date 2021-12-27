
import numpy as np 

import matplotlib.pyplot as plt 
import time
import pylab as pl
from IPython import display

class Perceptron:
    
    # input_size: dimension of the input including bias
    def __init__(self,input_size):
      
        # we store the input size because we will need it later
        self.input_size = input_size
        
        # weights (w) in randomly initalized to be the same size as the input
        self.w = np.random.randn(input_size,1).reshape(input_size,1)
        
        # we will store our accuracy after each iteration here
        self.history = []
        
    def train(self,X,Y, max_epochs = 100):
      
        # we clear history each time we start training
        self.history = []
        
        converged = False
        epochs = 0
        
        #set the learning rate 
        lr = 0.8

        while not converged and epochs < max_epochs :
            
            # TODO
            # 1. add training code here that updates self.w 
            # 2.  a criteria to set converged to True under the correct circumstances. 
            
            # curent strategy is random search (not good!)
            #self.w = np.random.randn(self.input_size,1)

            random_index = np.random.randint(len(X))
            random_X,random_Y = X[random_index].reshape(1,-1),Y[random_index].reshape(1,-1)    #reshape to has 2 dims

            if random_Y == 1 and random_X@self.w < 0:
                self.w += (random_X.T) * lr         #.T is the transpose of ndarray so that the size match 
            if random_Y == 0 and random_X@self.w >= 0:
                self.w -= (random_X.T) * lr
            
            
            # after training one epoch, we compute again the accuracy
            self.compute_train_accuracy(X,Y)
            epochs += 1 
            
        #if all correct, then converged
        if self.history[-1] == 1 : converged = True 
        
        if epochs == max_epochs:
          print("Qutting: Reached max iterations")
          
        if converged:
          print("Qutting: Converged")
          
        self.plot_training_history()
    
    def train3(self,X,Y,Z, max_epochs = 100):
      
        # we clear history each time we start training
        self.history = []
        
        converged = False
        epochs = 0
        
        #set the learning rate 
        lr = 0.8
        de = 0.1 #decay rate 
        st = 2  #step of decay 

        step = 1 
        while not converged and epochs < max_epochs :
            
            # TODO
            # 1. add training code here that updates self.w 
            # 2.  a criteria to set converged to True under the correct circumstances. 
            
            # curent strategy is random search (not good!)
            #self.w = np.random.randn(self.input_size,1)

            random_index = np.random.randint(len(X))
            random_X,random_Y,random_Z = X[random_index].reshape(1,-1),Y[random_index].reshape(1,-1),Z[random_index].reshape(1,-1)  #reshape to has 2 dims
            
#             if step%st == 0: 
#                 lr = lr*de
            if random_Y == 1:
                if self.w < 0: self.w = 0+lr*self.w
                if self.w > 0: self.w = 0.33-lr*self.w
                if self.w < 0: self.w = 0.66-lr*self.w
            
                  
            
            # after training one epoch, we compute again the accuracy
            self.compute_train_accuracy(X,Y)
            epochs += 1 
            
        #if all correct, then converged
        if self.history[-1] == 1 : converged = True 
        
        if epochs == max_epochs:
          print("Qutting: Reached max iterations")
          
        if converged:
          print("Qutting: Converged")
          
        self.plot_training_history()
    
    # The draw function plots all the points and our current estimate 
    # of the boundary between the two classes. Point are colored according to
    # the current output of the classifier. Ground truth boundary is also
    # plotted since we know how we generated the data
    
    def draw(self,X):
      
        pl.close()
        out = np.matmul(X,self.w).squeeze()
        
        P = X[out >= 0,:] 
        N = X[out.T < 0,:]
        
        x = np.linspace(0,1)
        
        pl.xlim((0,1))
        pl.ylim((0,1))
 
        pl.plot(P[:,0],P[:,1],'go', label = 'Positive')
        pl.plot(N[:,0],N[:,1],'rx', label = 'Negative')
        pl.plot(x, x, label = 'GT')
        
        a = self.w[0]
        b = self.w[1]
        c = self.w[2]
        
        pl.plot(x, -a/b * x - c/b, label = 'Estimated')
        
        pl.axis('tight')
        pl.legend()
        
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(0.1)
        
    
    # This computes the accuracy of our current estimate
    def compute_train_accuracy(self,X,Y):
        out = np.matmul(X,self.w)
        Y_bar = (out >= 0)
        accuracy = np.sum(Y==Y_bar)/np.float(Y_bar.shape[0])
        self.history.append(accuracy)
        print("Accuracy : %f " % (accuracy))
        self.draw(X)
        
    # Once training is done, we can plot the accuracy over time 
    def plot_training_history(self):
        #display.clear_output(wait=True)
        pl.close()
        plt.ylim((0,1.01))
        plt.plot(np.arange(len(self.history))+1, np.array(self.history),'-x')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
      