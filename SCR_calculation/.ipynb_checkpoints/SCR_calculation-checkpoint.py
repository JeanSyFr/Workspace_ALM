# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 10:52:39 2022

Calculate impact on SCR from a change of allocation. 
Needs following elements :
    - inputs from data_import.py
    - modele ML to estimate the own-funds and sensitities

@author: PICARDC
"""

import os
from sqlalchemy import false

# dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/home/jovyan/shared/ML_project_ALM'
os.chdir(dir_path)

import tensorflow as tf
import pandas as pd
import numpy as np
import data_import
from tf.keras.models import load_model
#import utilities
from train import evaluation
import dill


### Import scenarios ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
scenarios_dict  = data_import.import_scenarios_data(path_scenario = "Data//Q122//data_scr//scenarios")



### Import ML model to estimate CFs '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
path_model = ".//Weights"

# The name of the model loaded in this part, is consistent with what we trained and saved in the upper part
model_name = 'model_LSTM_v0'
model_filename = model_name + '.HDF5' 
model_data_filename = model_name + '_data_info.dill'

# Load the trained model and required model data
file_handle = open(path_model + "//" + model_data_filename, 'rb')
model_info = dill.load(file_handle)
file_handle.close()
#model_info = utilities.load_file(path_model + "//" + model_data_filename)
model = load_model(path_model + "//" + model_filename)
general_params, scenario_params, asset_params, corporate_params, output_params, liability_params= model_info['loading_params']
X_columns, scaler_x,  = model_info['X_ordered_cols'], model_info['scaler_X']



### Evaluation of ML model '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Preparing input data by reshaping and scaling based on the train scaler
mean_PV_oos = []
n_timesteps = 61
    
# loop on all scenarios
for  scenario_name in scenarios_dict.keys():    
    # Input data
    print(scenario_name)
    X_oos = scenarios_dict[scenario_name]
    # Data treatment
    extra_columns = list(set(X_columns) - set(X_oos.columns))
    n_instances_oos = int(X_oos.shape[0]/n_timesteps)    
    X_oos = X_oos[X_columns]   
    
    # Scaling inputs
    X_nn_oos = scaler_x.transform(X_oos.values).reshape(n_instances_oos, n_timesteps, X_oos.shape[1])
    
    # Make predictions
    scaled_predictions = model.predict(X_nn_oos).reshape(-1,1)
    
    # Scale the Y based on the y_scaler
    prediction_oos = model_info['y_pipeline'].inverse_transform(scaled_predictions, X_oos)
    
    # Convert the prediction array into a datafram
    mean_PV_oos.append(numpy.squeeze(evaluation.calculate_pv(prediction_oos, X_oos).mean().values))
        
mean_pv_cfs = pd.DataFrame(mean_PV_oos, columns = ['mean_pv_cfs'], index = input_dict.keys())



###  import data to calculate SCR impact ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# path
path_frp            = ".//Data//Q122//data_scr//frp

# data input from frp
corr_df             = pd.read_excel(path_frp + '//correlations_matrix.xlsx', index_col=0)    
standalone_scr_frp  = pd.read_excel(path_frp + '//standalones_scr_frp.xlsx',  index_col=0)
standalones_ref     = pd.read_excel(path_frp + '//standalones_scr_mac.xlsx',  index_col=0)
standalones_ref     = standalones_ref.drop(columns = ['risk_id'])
diversified_scr     = pd.read_excel(path_frp + '//diversified_scr.xlsx')


### standalone risk calculation using ML model '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# calculation of standalones using own-funds (reference to 1000 or 5000 simulations depending on risk)
new_market_stds     = mean_pv_cfs.copy()
new_market_stds     = new_market_stds[mean_pv_cfs.index.isin(standalones_ref.index)]
market_risks_name   = new_market_stds.index
risks_ref5000       = ['EQVol', 'IRVol', 'spread'] # risk factor with a reference of 5000 simulations
new_market_stds[new_market_stds.index.isin(risks_ref5000)] = mean_pv_cfs.filter(items = ['central_5000'], axis=0).values - new_market_stds[new_market_stds.index.isin(risks_ref5000)].values
new_market_stds[new_market_stds.index.isin(risks_ref5000) == False] = mean_pv_cfs.filter(items = ['central_1000'], axis=0).values - new_market_stds[new_market_stds.index.isin(risks_ref5000) == False].values

# concatenate standalones before and after on one dataframe
standalones_other   = standalones_ref.copy()
standalones_other.loc[market_risks_name,:] = new_market_stds.values
standalones         = pd.concat([standalones_ref, standalones_other], axis=1)
standalones.columns = ["Reference STD MAC", "ML STD MAC"]
del standalones_other

# taking worst IR risk between IRdown and IRup on ML estimation
if (standalones.loc["IRUp"]["ML STD MAC"] > standalones.loc["IRDown"]["ML STD MAC"] ):
    IR_value = standalones.loc["IRUp"].copy()
else:
    IR_value = standalones.loc["IRDown"].copy()

# taking worst IR risk between IRdown and IRup on references standalones MAC
if (standalones.loc["IRUp"]["Reference STD MAC"] < standalones.loc["IRDown"]["Reference STD MAC"] ):
    IR_value["Reference STD MAC"] = standalones.loc["IRDown"]["Reference STD MAC"]           
else:
    IR_value["Reference STD MAC"] = standalones.loc["IRUp"]["Reference STD MAC"]       

IR_value.name = "IR"        
standalones   = standalones.append(IR_value) 
standalones   = standalones.drop(["IRUp", "IRDown"])

# calculation of variations of standalones (in %)
standalones[ "variation" ] = (standalones["ML STD MAC"] / standalones["Reference STD MAC"]).replace(np.nan, 0)
standalones                = standalones.join(standalone_scr_frp)
standalones["impact_frp"]  = standalones["variation"] * standalones["frp_scr_hd"] -  standalones["frp_scr_hd"]


### calculation of SCR impact '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
standalone_index_list = list()

for standalone in standalones.index:
    standalone_index_list.append( int(np.where(standalone == standalone_scr_frp.index )[0]) )    


standalones          = standalones.sort_index(level = standalone_index_list)       
impact_frp           = standalones["variation"] * standalones["frp_scr_hd"]
unchanged_impact_frp = standalone_scr_frp.drop( standalone_scr_frp.index[ standalone_index_list ], axis=0 )["frp_scr_hd"]    
impact_frp_after     = pd.concat( [impact_frp, unchanged_impact_frp], axis = 0).reindex( standalone_scr_frp.index   )
impact_frp_before    = pd.concat( [standalones["frp_scr_hd"], unchanged_impact_frp], axis = 0).reindex( standalone_scr_frp.index   )


# VcV computation to get diverisified SCR (using correlation matrix)
VCV_before = np.sqrt ( impact_frp_before.T.dot( np.array( corr_df ).dot( impact_frp_before ) ) )
VCV_after  = np.sqrt ( impact_frp_after.T.dot( np.array( corr_df ).dot( impact_frp_after ) ) )

# Net impact 
capital_charge  = diversified_scr["capital_charge"][0]
VaRHD_995       = diversified_scr["VaR_HD_995"][0]
dt_mvbs         = diversified_scr["dt_mvbs"][0]
tax_rate        = diversified_scr["tax_rate"][0]
own_funds       = diversified_scr["own_funds"][0]
cross_effects   = diversified_scr["cross_effects"][0]

risk_capital_after_diversification_before = VaRHD_995
risk_capital_after_diversification_after  = VaRHD_995 + (VCV_after - VCV_before ) * (VaRHD_995 - cross_effects) / VCV_before

variation = 0 # no impact on OF for the moment

own_funds_before =  own_funds
own_funds_after  = own_funds_before *(1 + variation)

tax_relief_before = dt_mvbs
tax_relief_after  = dt_mvbs + tax_rate * (own_funds_before - own_funds_after )       

risk_capital_before =  capital_charge + tax_relief_before + risk_capital_after_diversification_before        
risk_capital_after  = risk_capital_before + (risk_capital_after_diversification_after - risk_capital_after_diversification_before) + (tax_relief_after - tax_relief_before)

impact = (risk_capital_after - risk_capital_before)/1000000

mean_pv_cfs.loc['impact_SCR_brut'] = risk_capital_after_diversification_after - risk_capital_after_diversification_before
mean_pv_cfs.loc['impact_SCR_net'] = impact

## Save results
path_output = ".//SCR_calculation//output"
mean_pv_cfs.to_excel(path_output + "//mean_pv_cfs_" + model_name + '.xls')

del mean_pv_cfs


