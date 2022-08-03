import tensorflow as tf
import numpy as np
import data_reader


class trainer(object):
    
    def __init__(self, models, reader, max_epochs=200, batch_size=None, batches = 15, validation_split=0.1, early_stopping = True):
        self.models = models
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.reader = reader
        if batch_size == None:
            self.batch_size = max(1,int(self.reader.x_train.shape[0]/batches))
        else:
            self.batch_size = batch_size
        
        if early_stopping:
            self.early_stopping =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
            self.callbacks = [self.early_stopping]
        else:
            self.callbacks = None
            
        
    def fit_models(self):
        
        for m in self.models: 
            np.random.seed(1)
            
            
            history = m._model.fit(self.reader.x_train, self.reader.y_train, epochs =self.max_epochs, batch_size = self.batch_size,validation_split = 0.2, shuffle = True, verbose = 1, callbacks = self.callbacks)
            
            m.train_hists.append(history)
                
        
        

        