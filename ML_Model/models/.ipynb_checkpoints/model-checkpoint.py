import tensorflow as tf
import numpy as np
#import ALM_ML_model as mdl
import models.my_layers as ml

from abc import ABC,abstractmethod # classes abstractions

metrics = [
    tf.keras.metrics.mean_absolute_error,
    tf.keras.metrics.mean_squared_error,
    tf.keras.metrics.mean_absolute_percentage_error
] 


class Model(ABC):
    ######################################################################################################
    # Creating abstract class for our models in order to specify the set of methods needed in all models #
    ###################################################################################################### 
    
    @abstractmethod
    def build_architecture(self):
        pass
    @abstractmethod
    def build_optimizer(self):
        pass
    @abstractmethod
    def compilation(self):
        pass

class LSTM_Model(Model):
    
    
    def __init__(self, hiden_layers_sizes = [100] ,years = 60, features_size = 14, simple_construct = True, lr = 0.001, decay = None, name= 'LSTM_'):
        """
        hiden_layers_sizes (Default value [100]) : A list of lstm_cells' sizes.
        years (Default value 60) : time-steps of the input.
        features_size (Default value 14) : numbers of features in each year.
        
        simple_construct (simple_construct = True) : Une seule couche LSTM contruite directement avec keras.LSTM
        
        lr (Default value 0.001) : learning rate for the optimizer.
        decay (Default value None) : decay of the learning rate for the optimizer.
        
        """
        
        #Training history of the model. We register each hsitory after fitting our model.
        self.train_hists = []
        # Name of our model according to LSTM depth
        self.name = name + str(len(hiden_layers_sizes))
        
        # Setting model's architecture's arameters #
        ############################################
        self._years = years
        self._features_size = features_size
        self._hiden_layers_sizes = hiden_layers_sizes
        

        # Setting learning (optimiser) parameters #
        ###########################################
        self.lr = lr
        self.decay = decay
        self.loss = lambda x, y : (2*metrics[1](x,y)+metrics[0](x,y)) #*metrics[2](x,y)
        self.validation_split = 0.1
        
        
        self.build_architecture(hiden_layers_sizes, simple_construct)
        self.build_optimizer()
        self.compilation()
        
        
            
    def build_architecture(self,hiden_layers_sizes,simple_construct):
        """
        Initializing the model using keras Sequential API
        """
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.Input(batch_input_shape=(None, self._years, self._features_size))) #shape=(self._years, self._features_size)
        
        ##self._model.add(tf.keras.layers.Dense(self._features_size*4, activation = 'linear'))
        self._model.add(ml.Dense_Id(self._features_size*5, activation = 'tanh'))
        #self._model.add(tf.keras.layers.Dense(self._features_size*5, activation = 'tanh'))
        #self._model.add(tf.keras.layers.Dense(self._features_size*5, activation = 'relu'))


        self.add_Deep_LSTM(hiden_layers_sizes, simple_construct)
            
        #Adding dense linear layer for regression "tf.nn.leaky_relu"
        self._model.add(tf.keras.layers.Dense(1, activation ='linear' ))
                 
    def add_Deep_LSTM(self, hiden_layers_sizes, simple_construct):
        """
        Creating the LSTM cell structure and connect it with the RNN structure #
        """
        if simple_construct:
            #Contruct one layer LSTM using compact LSTM keras-layer
            self._model.add(
                tf.keras.layers.LSTM(
                    self._hiden_layers_sizes[0],
                    return_sequences = True, 
                    activation = "tanh",
                    input_shape = (self._years, self._features_size)
                    )
            )
            return
        # Create the LSTM Cells. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument  of LSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        lstm_cells = []
        for n in hiden_layers_sizes:
            lstm_cells.append(tf.keras.layers.LSTMCell(n))
            

        # By taking in the LSTM cells as parameters, the StackedRNNCells function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of stacked simple cells.
        stacked_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)


        ########################################################################################################
        # Instantiating our RNN model and setting stateful to True to feed forward the state to the next layer #
        ########################################################################################################
        self._RNNlayer  =  tf.keras.layers.RNN(stacked_lstm, [None, self._years], trainable=True)

        """
        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = tf.Variable(tf.zeros([self.batch_size,self._features_size]),trainable=False)
        self._RNNlayer.inital_state = self._initial_state"""

        ############################################
        # Adding RNN layer to keras sequential API #
        ############################################        
        self._model.add(self._RNNlayer)
    
    def build_optimizer(self):
        """
        Instantiating the stochastic gradient decent optimizer
        """
        self._optimizer = tf.keras.optimizers.Adam( self.lr)
    
    def compilation(self):
        """
        Compiling and summarizing the model stacked using the keras sequential API
        """
        self._model.compile(loss= self.loss, optimizer=self._optimizer, metrics = metrics, sample_weight_mode = "temporal")
        self._model.summary()  