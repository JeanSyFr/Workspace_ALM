# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:36:50 2022

Import of data used to calculated an SCR impact :
        - scenarios (ESGs) : for central calculation (1000 and 5000) and each risk factor :  IntRatesDown, IntRatesUp, Equity, RealEstate, EquityVolUp, IRVolUp, Inflation
        - FRP inputs : 1in200 std by RF, total diversified SCR, Correlations Matrix
    
@author: PICARDC
"""
import os
import pandas as pd

def import_scenarios_data (path_scenario = "//.//Data//Q122//data_scr//scenarios"):
    '''
    Parameters
    ----------
    path_scenario : path to scenarios data. The default is ".//Data//Q122//data_scr//frp"

    Returns
    scenarios_dict : a dictionnaries with all all risk / scenarios tested, the scenarios in ML model format (input)
    '''
    
    # Name of scenarios
    scenario_param = pd.read_excel(path_scenario  + '//scenario_param_cloud.xlsx')
    name_list      = scenario_param['risk_factor'].tolist()

    ## Prepare the inputs and import scenarios data as dictionnary of processed scenarios
    scenarios_list = []
    
    for i in range(0, scenario_param.shape[0]) :
        print(i)
        scenarios = pd.read_csv(scenario_param.loc[i, 'scenario_filename'], index_col=['Scenario', 'Timestep'])
        scenarios_list.append(scenarios)
    
    scenarios_dict = dict(zip(name_list, scenarios_list))
    
    del name_list, scenarios_list , scenario_params_list, scenarios
    
    
    ## Clean memory
    import gc
    gc.collect()
    
    return scenarios_dict
    



