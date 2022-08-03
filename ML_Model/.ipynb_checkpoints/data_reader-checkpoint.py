"""
reader
# TODO : 
     - Importer
     - Scalling Input by features
     - Scalling Output by Year
     - Splitting
     """
import pandas as pd
import numpy as np





class reader(object):
    
    
    def __init__(self, data_path, scal_target = True, test_split_rate = 0., year_0 = True, spread = False):
        
        
        self.data_name = data_path
        self.test_split_rate = test_split_rate
        
        
        
        X = pd.read_csv("../"+data_path+"/input.csv")
        Y = pd.read_csv("../"+data_path+"/output_ALM.csv")
        
        scenarios = int(data_path.split("/")[-1].split("_")[0])
        years = X.iloc[-1,1]
        features_size = len(X.columns)-2
        
        x_shape = (scenarios, years + year_0, features_size)
        y_shape = (scenarios, years + year_0, 1)
        print("x_shape", x_shape)
        print("y_shape", y_shape)

    
        
        # Discarding scenraios and years columns
        x = np.reshape(np.array(X.iloc[:,2:], dtype = 'float32'), x_shape)
        y = np.reshape(np.array(Y.iloc[:,2:], dtype = 'float32'), y_shape)
        
        # remove the year 0 if needed
        if year_0:
            x = x[:,1:,:]
            y = y[:,1:,:]
        
        
        self.x_train = np.zeros(shape = x.shape)
        self.y_train = np.zeros(shape = y.shape)
        self.x_test = None
        self.y_test = None
        
        
        for feature in range(x.shape[-1] - (1-spread)):
            for year in range(x.shape[-2]):
                self.x_train[:,year,feature] = x[:,year,feature] - np.mean(x[:,year,feature])
                if np.std(self.x_train[:,year,feature]) < 0.001:
                    print(year, feature, np.std(self.x_train[:,year,feature]) )
                self.x_train[:,year,feature] /= np.std(self.x_train[:,year,feature])
        
        
            
        self.y_mean = None
        self.y_std = None
        
        if scal_target:
            self.y_mean = np.zeros(60)
            self.y_std = np.ones(60)
            for feature in range(y.shape[-1]):
                for year in range(y.shape[-2]):
                    self.y_mean[year] = np.mean(y[:,year,feature])
                    self.y_std[year] = np.std(y[:,year,feature])
                    
                    self.y_train[:,year,feature] = y[:,year,feature] - self.y_mean[year]
                    self.y_train[:,year,feature] /= self.y_std[year]
        else:
            self.y_train = y
            
        if test_split_rate !=0:
            np.random.seed(123)
            shuffled_scenario = np.arange(scenarios)
            np.random.shuffle(shuffled_scenario)
            end_train = int((1-test_split_rate)*scenarios)
            
            self.x_test = self.x_train[shuffled_scenario[end_train:],:,:]
            self.y_test = self.y_train[shuffled_scenario[end_train:],:,:]
            
            self.x_train = self.x_train[shuffled_scenario[:end_train],:,:]
            self.y_train = self.y_train[shuffled_scenario[:end_train],:,:]

            
    def descale_output(self, y_predicted):
        for feature in range(y_predicted.shape[-1]):
                for year in range(y_predicted.shape[-2]):
                    
                    y_predicted[:,year,feature] *= self.y_std[year]
                    y_predicted[:,year,feature] += self.y_mean[year]
        return y_predicted.copy()
    
    def scale_ouput(self, y):
        y_mean = np.zeros(60)
        y_std = np.ones(60)
        for feature in range(y.shape[-1]):
            for year in range(y.shape[-2]):
                y_mean[year] = np.mean(y[:,year,feature])
                y_std[year] = np.std(y[:,year,feature])
                    
                self.y_train[:,year,feature] = y[:,year,feature] - y_mean[year]
                self.y_train[:,year,feature] /= y_std[year]
        return y_mean, y_std
        
                    
                    
                    
        
        
            

        
        
        
        
    

