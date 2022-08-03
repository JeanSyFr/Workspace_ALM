'''
Evaluate an existing model
'''

# General
import copy
import numpy
import pandas

from . import helper_functions


def prepare_evaluation_table(inputs, target, predictions,
                             target_col=None, drop_zero=True,
                             discount_factors=None):
    """
        This function processes data for evaluating a given model.

        Arguments:
            inputs: Input data array (array or df).
            target: The ground truth (df).
            predictions: Model predictions (2D array).
            target_col: The name of the column in the target to be used (string or None).
                if None chosen then it requires the target to have just one column, which is the one that
                will be used.
            drop_zero: Whether to drop the zero timestep from the final table (True/False).
            discount_factors: Can pass discount_factors optionally to avoid re-calculating them,
            in which case inputs are ignored (array).

        Return:
            evaluation_table: A dataFrame with three columns: target, prediction, and discount_factor
            with the same index as target (df).
    """

    assert isinstance(target, pandas.DataFrame), 'Target must be a DataFrame'
    if target_col is None:
        assert len(
            target.columns) == 1, 'Target must have only one column if target_col not passed'

    if discount_factors is None:
        assert len(inputs.shape) == 2, 'Inputs must be 2D'

        # Calculate discount factors
        discount_factors = helper_functions.calculate_discount_factors(
            inputs).reshape(-1, 1)

    # Initialize the evaluation table to be a DataFrame of the same structure of the target
    if target_col is None:
        evaluation_table = target.copy()
    else:
        evaluation_table = target[[target_col]].copy()
        # The below will be used for selectiong a column based on position
        # Here target_col is integer
        # evaluation_table = target.iloc[:,[target_col] ].copy()

    # Rename the target column
    evaluation_table.columns = ['Target']

    # Add the predictions and discount_factors
    evaluation_table['Prediction'] = predictions
    evaluation_table['Discount factor'] = discount_factors.reshape(-1)

    if drop_zero:
        # First make sure that the last index is Timestep
        assert evaluation_table.index.levels[-1].name == 'Timestep'

        # Check if the first timestep is zero
        if evaluation_table.index.levels[-1][0] == 0:
            # Drop timestep 0
            if len(evaluation_table.index.levels) == 2:
                evaluation_table = evaluation_table.loc[pandas.IndexSlice[:, 1:], :]
            elif len(evaluation_table.index.levels) == 3:
                evaluation_table = evaluation_table.loc[pandas.IndexSlice[:, :, 1:], :]

            # Update the indices appropriately
            evaluation_table.index = evaluation_table.index.remove_unused_levels()

    return evaluation_table


def calculate_errors(evaluation_table, mode='all_errors'):
    """
        This function calculates the discounted error and optionally aggregates based on the mode
        specified.

        Arguments:
            evaluation_table: DataFrame with three columns: target, prediction, and discount_factor
            (df).
            mode: how to aggregate the errors. 'Strategy', 'Scenario', 'Timestep', 'all_errors' (str).

        Return:
            scenario_error: the absolute discounted residual error for each scenario
                (pd).
    """

    assert isinstance(evaluation_table,
                      pandas.DataFrame), "evaluation_table must be a pandas.DataFrame."
    assert isinstance(evaluation_table.index,
                      pandas.MultiIndex), "evaluation_table.index must be a MultIndex."

    # Get the level names
    level_names = set([level.name for level in evaluation_table.index.levels])
    # Make sure that mode is a valid level name
    # Note: our convention is that levels are named with the first letter being capital
    assert mode == 'all_errors' or mode in level_names,\
    "Invalid mode. Allowed mode values are 'all_errors' or one of the level names: %s"% str(level_names)

    # Calculate the error for each prediction
    all_errors = (evaluation_table['Target'] - evaluation_table['Prediction']
                 ).abs() * evaluation_table['Discount factor']

    if mode != 'all_errors':
        # Aggregate by this level
        errors = all_errors.groupby(level=mode.title()).mean()
    else:
        # No aggregation to be done
        errors = all_errors

    return errors

def calculate_pv_error(evaluation_table):
    """
        This function calculates the PV error. For each scenario, we calculate the predicted PV and real PV as well as the error between them. We will report pv error both in an absolute and relative errors format.

        Argument:
            evaluation_table: DataFrame with three columns: target, prediction, and discount_factor (df).
    """
    group_by_cols = (list(evaluation_table.index.names))[:-1]
    pv_actual = (evaluation_table['Target'] * evaluation_table['Discount factor']).groupby(group_by_cols).sum()
    pv_predicted = (evaluation_table['Prediction'] * evaluation_table['Discount factor']).groupby(group_by_cols).sum()
    pv_error = (pv_actual - pv_predicted)

    pv_error_ratio = (pv_error/pv_actual).abs()*100

    return pv_error, pv_error_ratio

def calculate_pv(y_original, X):
    """
        This function claculates the Present Value.

        Arguments:
            X: Input, needed to calculate discount factors (df).
            y_original: Output, it can be prediction or target value (array/df).
            train_on_DF: Wether the model is trained on cashflow or discounted cashflow. if y_original then train_on_DF is always False (True/False).

        Return:
            PV: The present values based on the y_original (df).
    """

    y = copy.deepcopy(y_original)
    if isinstance(y, numpy.ndarray):
        y = pandas.DataFrame(y, index = X.index, columns = ['y'])

    group_by_cols = (list(y.index.names))[:-1]


    discount_factors = helper_functions.calculate_discount_factors(X).reshape(-1, 1)
    y_df = y * discount_factors
    PV = y_df.groupby(group_by_cols).sum()
    PV.columns = ['PV']
    return PV














