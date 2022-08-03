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