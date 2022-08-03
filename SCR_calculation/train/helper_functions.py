# Imports
import pandas
import numpy
import time
import datetime
import os
import logging
import psutil
import sys
import copy
from sklearn.base import BaseEstimator, TransformerMixin

from . import evaluation



def log_text(message):
    """
        This function will print a given message in the log file.

        Argumnet:
            message: The message that we want to write into the log file (str).
    """
    try:
        # Add the handler
        logging.getLogger().addHandler(fh)
        # Write the messege in the log file
        logging.info(message)
        # Remove the handler and close that
        logging.getLogger().removeHandler(fh)
        fh.close()
    except: # The file is not created
        pass



def log_memory(message):
    """
        This function prints the memory usage info into a log file.

        Argumnets:
            message: The message that shows the status of the program (str).
    """
    try:
        # Add the handler
        logging.getLogger().addHandler(fh)
        # Write the messege in it
        logging.debug("-" * 10)
        logging.debug(message)
        logging.debug("available memory on machine: %s GB"\
                %(round(psutil.virtual_memory().available/float(10**9), 3)))
        logging.debug("non-swapped physical memory used by Python: %s GB"\
                     %(round(psutil.Process().memory_info().rss/float(10**9), 3)))
        # Remove the handler and close that
        logging.getLogger().removeHandler(fh)
        fh.close()
    except:  # In case the folder is not created
        pass


def create_log_file(project_path, run_name = '_'):
    """
        This function creates a log file to monitor the memory usage.
        Note: The name of the log file will be the actual date/time when this function is called.

        Argument:
            project_path: A log folder will be created within project path (str).
    """

    # Make the path of the log file
    ts = time.time()
    log_time =str( datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H')) + '.log'
    log_file_name = run_name + "_" + log_time
    log_folder = os.path.join(project_path,'logs')
    log_file_path = os.path.join(log_folder, log_file_name)

    # Make directory for the log file if it does not already exist
    try:
        os.mkdir(log_folder)
    except:
        pass

    try:
        os.remove(log_file_path )
    except FileNotFoundError:
        pass
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # Define a global varible shared among all functions calling that
    global fh
    fh = logging.FileHandler(filename=log_file_path)
    fh.setLevel(logging.DEBUG)


    log_text("Project path: %s" %project_path)
    log_text("helper_function_path: %s" %(os.path.abspath( __file__ )))
    log_text("evaluation path inside helper function: %s" %(os.path.abspath( evaluation.__file__ )) )


def load_data(pathname, index_col= ['Scenario', 'Timestep'], n_timesteps = None, index_name = None,
              cols_to_keep = None, cols_to_drop = None, drop_constant = False, verbose = False):
    """
        This function loads data (input/output) from a csv file.
        TODO: Function has to be generalized to other phases of the project.

        Arguments:
            pathname: The path of the data (str).
            index_col: List of columns from input dataset to set as index (list of str).
            n_timesteps: Number of timesteps to keep (int).  If not given, all timesteps will be kept.
            index_name: New names of the index columns (list of str). If not given, the index_cols will be used.
            cols_to_keep: List of columns to keep (list of str).
            cols_to_drop: List of columns to drop (list of str).
                It is possible to also give part of the names of that columns. e.g. all columns with 'USD' will be removed.
            drop_constant: Whether or not to remove the constant columns (True/False).
            verbose: Whether or not to show the information about data (True/False).

        Return:
            df: A multi-index dataframe (df).
    """

    # Get the memory before loading data
    log_memory('Memory usage before loading data')

    df = pandas.read_csv(pathname, index_col=index_col)

    if index_name is not None:
        # Rename the index names if new names are given
        df.index.names = index_name

    # Set the number of timesteps
    if n_timesteps is None:
        # Keep all the existing timesteps if n_timestep is not given
        n_timesteps = df.index.levels[-1].max()
    if df.index.nlevels == 2:
        df = df.loc[pandas.IndexSlice[:, 0: n_timesteps], :]
    elif df.index.nlevels == 3:
        df = df.loc[pandas.IndexSlice[:, :, 0:n_timesteps], :]
    df.index = df.index.remove_unused_levels()

    # Keep the necessary columns (for multi outputs)
    if  cols_to_keep is not None:
        assert isinstance(cols_to_keep, list)
        df = pandas.DataFrame(df[cols_to_keep])


    if cols_to_drop is not None:
        assert isinstance(cols_to_drop, list)
         # Get the full list of columns that we wanna remove
        cols_drop = []
        for keyword in cols_to_drop:
            cols_drop += [col for col in df.columns if keyword.lower() in col.lower()]
        df.drop(cols_drop, axis=1, inplace = True)
        if verbose:
            print ('Removed {} column(s) based on user request' .format(len(cols_drop)) )


    if drop_constant:
        # Drop constant columns
        n_cols_old = len(df.columns)
        df = df.loc[:, (df != df.iloc[0]).any()]
        n_cols_new = len(df.columns)
        if verbose:
            print('Removed {} constant column(s) from data'.format(n_cols_old - n_cols_new))

    if verbose:
        print('Data Information')
        for level in df.index.levels:
            print('Number of {}: {}'.format(level.name, len(level)))
        print('Number of column(s): {}'.format(len(df.columns)))
        print ('-'*30)

    # Get the memory before loading data
    try:
        log_memory('Memory usage after loading data')
    except:
        # For the OOS which we do not need log file
        pass

    return df

def custom_train_test_split(dfs, train_fraction = .5, shuffle = True, n_splits = 1):

    """
        For a given dataframe, divide it into test and train sets.

        Arguments:
            dfs: List of dataframe (list of dfs). Indices of all levels have to be exactly the same for all dataframes.
            train_fraction: Fraction of data for training (float).
            shuffle: Whether or not to shuffle the data (True/False).
            n_splits: Number of splits, n_split > 1 will be used for cross validation (int).

        Returns:
            splitted_data: List of tuples of the same length as dfs, each tuple is of the form of df_train, df_test.
            The returned dataframes are expected to be MultiIndex with the
            levels being one of the following possibilities:
            1. ['Scenario', 'Timestep']
            2. ['Strategy', 'Timestep']
            3. ['Strategy', 'Scenario', 'Timestep'].
    """

    log_memory("Memory usage before splitting data")

    # Step 0: Check to see if the general structure of the dataframes are correct
    for df in dfs:

        if df is None:
            continue
        assert isinstance(df, (pandas.DataFrame, pandas.Series))
        assert isinstance(df.index, pandas.MultiIndex)
        assert 2 <= len(df.index.levels) <= 3
        assert df.index.levels[-1].name == 'Timestep'

        # Check that the indices of all levels in all dfs match
        for i in range(df.index.nlevels):
            try:
                assert(df.index.levels[i] == dfs[0].index.levels[i]).all()
            except:
                raise ValueError('Indices of DataFrames do not match')

    # Step 1: If train_fraction is 1, we assign all the data to train and None to test set.
    if train_fraction == 1:
        print("Training will be done on the whole dataset!")
        splits = []
        for df in dfs:
            df_train, df_test = df, None
            splits.append((df_train, df_test))
        return splits

    # Step 2: Get the indices of all levels of one of the dataframes except the last index (i.e. Timestep).
    all_index_levels = [list(dfs[0].index.levels[i]) for i in range(dfs[0].index.nlevels -1)]

    # Step 3: Split all indices in all levels into train and test indices
    split_indices = train_test_split(all_index_levels, train_fraction=train_fraction,
                                     n_splits=n_splits, shuffle=shuffle)

    if 'Strategy' in dfs[0].index.names and 'Scenario' not in dfs[0].index.names \
     and len(split_indices) == 1:
        # Print ("Indices: Strategy, Timestep")
        train_strategy_indices, test_strategy_indices = split_indices[0]
        train_scenario_indices, test_scenario_indices = None, None
    elif 'Strategy' not in dfs[0].index.names and 'Scenario'  in dfs[0].index.names\
     and len(split_indices) == 1:
        # Print ("Indices: Scenario, Timestep")
        train_scenario_indices, test_scenario_indices = split_indices[0]
        train_strategy_indices, test_strategy_indices = None, None
    elif 'Strategy' in dfs[0].index.names and 'Scenario' in dfs[0].index.names\
     and len(split_indices) == 2:
        # Print ("Indices: Strategy, Scenario, Timestep")
        train_strategy_indices, test_strategy_indices = split_indices[0]
        train_scenario_indices, test_scenario_indices = split_indices[1]
    else:
        print("Invalid indices")

    # Step 4: Split dfs into test and train based on the indices from the last step
    splitted_data = perform_train_test_split_dfs(dfs,
                                                 train_strategy_indices, test_strategy_indices,
                                                 train_scenario_indices, test_scenario_indices)

    log_memory("Memory usage after splitting data")

    return splitted_data

def train_test_split(all_index_levels, train_fraction, shuffle = True, n_splits = 1):
    """
        Given the level of indices, this function splits them into train and test independently.
        Note: Assumes that the indices are numbered from 1 to index_length.

        Arguments:
            all_index_levels: List of levels of each index to be splitted on (list of list of int).
            train_fraction: The fraction of each level to be assigned to train (float).
            n_splits: Number of splits (int).

        Returns:
            all_train_test_indices: List of tuples, each tuple corresponds to an index level. Each tupple is
                                    a split of the form (train_indices, test_indices), where each of those
                                    elements is a list or array of the indices (int). (if n_splits == 1).
            all_train_test_indices: Non-modified all_train_test_indices if n_splits > 1.
    """

    assert 0 < train_fraction < 1
    assert n_splits >= 1

    all_train_test_indices = []
    for _ in range(n_splits):
        # Split each index
        train_test_indices = []
        for index_levels in all_index_levels:
            # Empty indices are not allowed
            index_length = len(index_levels)
            assert index_length > 0

            # Shuffle the indices
            if shuffle:
                numpy.random.shuffle(index_levels)

            # Find the number of strategies to have in the training
            num_indices_train = int(index_length * train_fraction)

            # Split the indices
            train_indices = index_levels[:num_indices_train]
            test_indices = index_levels[num_indices_train:]

            # Add this split to the list of all splits
            train_test_indices.append((train_indices, test_indices))

            # Make sure all indices have been assigned
            assert len(train_indices) + len(test_indices) == index_length

        all_train_test_indices.append(train_test_indices)

    # For the case of n_splits = 1, return the split instead of a list of splits
    if len(all_train_test_indices) == 1:
        all_train_test_indices = all_train_test_indices[0]

    return all_train_test_indices

def perform_train_test_split_dfs(dfs, train_strategy_indices=None, test_strategy_indices=None,
                                 train_scenario_indices=None, test_scenario_indices=None):
    """
        Perform a train/test split on a list of dataframes. The dataframes have to be in a specefic format detailed below.

        Arguments:
            dfs: List of dataframes to split (list of df). For convenience, None can be passed, but will be ignored.
            train_strategy_indices: Strategy indices to include in the train set (list of int).
            test_strategy_indices: Strategy indices to include in the test set (list of int).
            train_scenario_indices: Scenario indices to include in the train set (list of int).
            test_scenario_indices: Scenario indices to include in the test set (list of int).

        Returns:
            List of tuples of same length as dfs, each tuple is of the form df_train, df_test.
            The dataframes in df are expected to have a MultiIndex with the levels being one
            of the following possibilities:
                1. ['Scenario', 'Timestep']
                2. ['Strategy', 'Timestep']
                3. ['Strategy', 'Scenario', 'Timestep'].
    """

    # Convenient shorthand notation for slicing an index
    idx = pandas.IndexSlice
    splits = []
    for df in dfs:

        # Check if there is a strategy index
        try:
            strategy_index = df.index.names.index('Strategy')
        except ValueError:
            strategy_index = -1

        # Check if there is a scenario index
        try:
            scenario_index = df.index.names.index('Scenario')
        except ValueError:
            scenario_index = -1

        if len(df.index.levels) == 2:

            # There are only two possibilities for the first index
            if strategy_index == 0:
                df_train = df.loc[idx[train_strategy_indices, :]]
                df_test = df.loc[idx[test_strategy_indices, :]]
            elif scenario_index == 0:
                df_train = df.loc[idx[train_scenario_indices, :]]
                df_test = df.loc[idx[test_scenario_indices, :]]
            else:
                raise ValueError("No index to split on,\
                expected first index of the two to be Strategy or Scenario")

        if len(df.index.levels) == 3:
            # There is only one possiblity
            if strategy_index == 0 and scenario_index == 1:
                df_train = df.loc[idx[train_strategy_indices,
                                      train_scenario_indices, :]]
                df_test = df.loc[idx[test_strategy_indices,
                                     test_scenario_indices, :]]
            else:
                raise ValueError("No index to split on, \
                expected first index of the three to be Strategy and the next to be Scenario")

        # Remove levels that are unused due to the slicing
        df_train.index = df_train.index.remove_unused_levels()
        df_test.index = df_test.index.remove_unused_levels()

        splits.append((df_train, df_test))

    return splits


def calculate_discount_factors(X, return_array=True, column_name='EUR.ZeroCouponBondPrice(1y)'):
    """
        This function calculates the discount factors implied by the EURO bond prices and returns list of discount factors, given X.

        Arguments:
            X: A MultiIndex dataframe containing the column 'EUR.ZeroCouponBondPrice(1y)'
               with the last index being Timestep (Timestep 0 included) (df).
            return_array: Whether to return an array or a pandas.Series (True/False).

        Returns:
            discount_factors: Either a 2D array with the second dimension being num_timesteps,
                              or a pandas.Series, see the return_array flag above (df).
    """

    assert isinstance(X, pandas.DataFrame),\
    "X must be a DataFrame for calculate_discount_factors to work"

    # Assert 'EUR.ZeroCouponBondPrice(1y)' in X.columns,
    assert X.index.levels[-1].name == 'Timestep',\
    "X must have Timestep as the last index"
    assert min(X.index.levels[-1]) == 0,\
    "Timestep 0 must be included for the discount factors to be calculated properly"

    # Get index names
    levels = [level.name for level in X.index.levels]

    # Perform a cumulative sum on each scenario separately. We do it by grouping all levels except Timestep.
    discount_factors = X[column_name].groupby(
        levels[:-1]).cumprod()
    discount_factors.name = 'discount_factors'

    # Shift down discount_factors by one timestep, for each scenarios
    discount_factors = discount_factors.groupby(levels[:-1]).shift()

    # Set the discount factor for timestep 0 to 1
    if len(levels) == 2:
        discount_factors.loc[pandas.IndexSlice[:, 0]] = 1
    elif len(levels) == 3:
        discount_factors.loc[pandas.IndexSlice[:, :, 0]] = 1
    else:
        raise ValueError(
            "calculate_discount_factors only works for dataframes with a MultiIndex of 2 or 3")

    # Check that the index matches the actual levels used
    if return_array:
        # Make into a 2D array with num_timesteps as the second dimension
        num_timesteps = len(discount_factors.index.levels[-1])
        discount_factors = discount_factors.values.reshape(-1, num_timesteps)

    return discount_factors



def calculate_discount_factors_59(X, return_array=True, column_name='EUR.ZeroCouponBondPrice(1y)'):
    """
        This function calculates the discount factors implied by the EURO bond prices and returns list of discount factors, given X.

        Arguments:
            X: A MultiIndex dataframe containing the column 'EUR.ZeroCouponBondPrice(1y)'
               with the last index being Timestep (Timestep 0 included) (df).
            return_array: Whether to return an array or a pandas.Series (True/False).

        Returns:
            discount_factors: Either a 2D array with the second dimension being num_timesteps,
                              or a pandas.Series, see the return_array flag above (df).
    """

    assert isinstance(X, pandas.DataFrame),\
    "X must be a DataFrame for calculate_discount_factors to work"

    # Assert 'EUR.ZeroCouponBondPrice(1y)' in X.columns,
    assert X.index.levels[-1].name == 'Timestep',\
    "X must have Timestep as the last index"
    assert min(X.index.levels[-1]) == 0,\
    "Timestep 0 must be included for the discount factors to be calculated properly"

    # Get index names
    levels = [level.name for level in X.index.levels]

    # Perform a cumulative sum on each scenario separately. We do it by grouping all levels except Timestep.
    discount_factors = X[column_name].groupby(
        levels[:-1]).cumprod()
    discount_factors.name = 'discount_factors'

    # Shift down discount_factors by one timestep, for each scenarios
    discount_factors = discount_factors.groupby(levels[:-1]).shift()

    # Set the discount factor for timestep 0 to 1
    if len(levels) == 2:
        discount_factors.loc[pandas.IndexSlice[:, 0]] = 1
    elif len(levels) == 3:
        discount_factors.loc[pandas.IndexSlice[:, :, 0]] = 1
    else:
        raise ValueError(
            "calculate_discount_factors only works for dataframes with a MultiIndex of 2 or 3")

    # Check that the index matches the actual levels used
    if return_array:
        # Make into a 2D array with num_timesteps as the second dimension
        num_timesteps = len(discount_factors.index.levels[-1])
        num_timesteps = 60
        discount_factors = discount_factors.values.reshape(-1, num_timesteps)

    return discount_factors






























def mean_absolute_error(predictions, Y, sample_weight):
    """
        This function calculates the mean of the weighted absolute error in the predictions.

        Arguments:
            predictions: Predictions of model (array).
            Y: Actual target values (df).
            sample_weight: Weights for each target value (array).

        Return:
            The mean weighted absolute error of the predictions (array).
    """

    assert Y.shape == predictions.shape, "predictions and target must be of same shape"

    # Preparation step: construct an evaluation table
    evaluation_table = evaluation.prepare_evaluation_table(
        inputs=None, target=Y, predictions=predictions, discount_factors=sample_weight)

    # Calculate the prediction error for each example
    prediction_errors = evaluation.calculate_errors(
        evaluation_table)

    # Calculate the mean
    return numpy.mean(prediction_errors)




def expand_param_grid(model_name, param_grid):
    """
        Expand a param_grid to contain the model name in the param_name as required for pipelines.

        Arguments:
            model_name: Name of model (string).
            param_grid: A dict from param_name to param_values (dict).

        Return:
        new_param_grid: New param_grid with the model's name as a prefix for each param_name (dict).
    """

    if param_grid == {}:
        return {}

    # Create the new param grid
    new_param_grid = {}
    for key, value in param_grid.items():
        new_param_grid[model_name + '__' + key] = value

    return new_param_grid


class constant_scaler(BaseEstimator, TransformerMixin):
    """
        Scaler which shrinks target by the magnitude of the shrinkage rate.
    """

    def __init__(self, shrinkage_rate=1e6):
        ''' Arguments:
            shrinkage_rate : rate to devide data by (default = 1e6)
        '''
        self.shrinkage_rate = shrinkage_rate

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            return X/float(self.shrinkage_rate).values
        except:
            return X/float(self.shrinkage_rate)


    def inverse_transform(self, X, y=None):

        return X*(float(self.shrinkage_rate))




class y_pipeline(BaseEstimator, TransformerMixin):
    '''
    A class for transformin and scaling output and to change it back
    Attributes:
        scaler: Scaler for output
        mode_train: Transformation applied on the output, three attributes are accepted
            1."CF": Cash Flow, original data
            2."DCF": Discounted cash flow
            3."cumulative": cumulative of discounted cashflow
    methodes:
        transform
        inverse_transform
    Return:
        for transform it will return processed output and weights
        for the inverse_transfrom will return the unscaled and reveresed transformed of output.
    '''

    def __init__(self, scaler = None, mode_train = "CF"):
        self.scaler = scaler
        self.mode_train = mode_train
    def fit(self, y, X= None):
        return self

    def transform(self, y, X= None):
        '''
        transorm y (if mode_train is DCF or cumulative)and scale it
        Arguments:
            y : output (pandas, DataFrame)
            X : input (it is needed for calculation of discounted factors)
        '''
        assert isinstance(y, pandas.DataFrame), "the input (y) should be a data frame"
        assert self.mode_train is "CF" or "DCF"or "cumulative", "value for train_mode should be one of these values: CF, DCF, cumulative"
        self.n_timesteps  =  len(y.index.levels[-1])
        assert X is not None, "X is needed to calculate the discounted cash flow"

        if self.mode_train != "CF":
            assert len(X) == len(y), "length of X and y should be equal"
            discount_factors = calculate_discount_factors(X, return_array = False).values.reshape(-1,1)
            y_t = y * discount_factors
            weights = numpy.ones((int(len(y)/self.n_timesteps), self.n_timesteps))
        else:
            y_t = y
            weights = calculate_discount_factors(X, return_array = True)

        if self.mode_train == "cumulative":
            y_t = y_t.groupby( list(y.index.names)[:-1] ).cumsum()

        if self.scaler is not None:
            try :
                y_t = self.scaler.transform(y_t.values.reshape(-1, self.n_timesteps)).reshape(-1,1)

            except:
                y_t = self.scaler.fit_transform(y_t.values.reshape(-1, self.n_timesteps)).reshape(-1,1)

        return [y_t, weights]


    def inverse_transform(self, y, X=None):
        '''
        inverse transorm y (iverse scale and inverse tranform)
        Arguments:
            y : output (pandas, DataFrame or numpy.array)
            X : input (it is needed for calculation of discounted factors)
        '''

        if self.scaler is not None:
            try:
                y_t = self.scaler.inverse_transform(y.reshape(-1, self.n_timesteps)).reshape(-1,1)
            except :
                y_t = self.scaler.inverse_transform(y.values.reshape(-1, self.n_timesteps)).reshape(-1,1)

        else :
            y_t = y

        if self.mode_train != "CF":

            assert X is not None, "X is needed to calculate the discounted cash flow"
            assert len(X) == len(y), "length of X and y should be equal"
            discount_factors = calculate_discount_factors(X, return_array = False).values.reshape(-1,1)

            if self.mode_train == "cumulative":

                y_t_pd = pandas.DataFrame(y_t, index = X.index)
                y_t = (y_t_pd.groupby(list(y_t_pd.index.names)[:-1]).diff().fillna(0)).values

            y_t = y_t/ discount_factors

        return y_t
