# from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import tempfile
import json
import pandas as pd
import yaml
import pickle
from datetime import datetime

import numpy as np
from joblib import load
import tensorflow as tf
import os
import requests


def load_config_with_overrides(user_overrides=None):
    with open('cbm.yaml','r') as file:
        utility_dict = yaml.safe_load(file)
    
    if user_overrides:
        if 'efd_features' in user_overrides:
            utility_dict['efd_features'] = user_overrides['efd_features']
        if 'fault_matrix' in user_overrides:
            fault_matrix_df = create_fault_matrix_from_frontend(user_overrides['fault_matrix'])
            temp_excel_path = './temp_fault_matrix.xlsx'
            fault_matrix_df.to_excel(temp_excel_path, index=False)
            utility_dict['f_mat_path'] = temp_excel_path
    
    return utility_dict

# For now, load default config
utility_dict = load_config_with_overrides()

def create_fault_matrix_from_frontend(fault_data):
    """Convert frontend fault matrix JSON to Excel format with strict validation"""
    allowed_params = ['Texh', 'Pscav', 'Ntc', 'Pcomp_Pscav', 'Pmax', 'Pcomp', 'PR', 'Ntc_Pscav']
    
    # Map frontend names to internal names
    param_mapping = {
        'Texh': 'Texh',
        'Pscav': 'Pscav', 
        'NTC': 'Ntc',
        'Pcomp / Pscav': 'Pcomp_Pscav',
        'Pmax': 'Pmax',
        'Pcomp': 'Pcomp',
        'PR': 'PR',
        'NTC / Pscav': 'Ntc_Pscav'
    }
    
    rows = []
    
    # Extract fault data from the new JSON format
    fault_values = fault_data.get('values', {})
    
    for fault_name, fault_info in fault_values.items():
        # Validate dominant parameter
        dominant_raw = fault_info.get('Dominant Parameter', 'pmax').lower()
        dominant_map = {'pmax': 'Pmax', 'pcomp': 'Pcomp', 'texh': 'Texh'}
        dominant = dominant_map.get(dominant_raw, 'Pmax')
        
        # Auto-generate Fault_id
        # fault_id = fault_name.replace(' ', '').replace('-', '').replace('_', '')[:12]
        fault_id_mapping = {
        'Injection system fault': 'InjSysFault',
        'Start of injection too late': 'StaInjLate', 
        'Start of injection too early': 'StaInjEarly',
        'Exhaust valve leaking': 'ExhValvLeak',
        'Blow-by in combustion chamber': 'BloCombChabr',
        'Exhaust valve early opening': 'ExhValEarOpn',
        'Exhaust valve late opening': 'ExhValLatOpn',
        'Exhaust valve early closure': 'ExhValEarlClos',
        'Exhaust valve late closure': 'ExhValLatClos'
        }
        fault_id = fault_id_mapping.get(fault_name, fault_name.replace(' ', '').replace('-', '').replace('_', '')[:12])
        
        # Build row
        row = {
            'Fault_id': fault_id,
            'Fault': fault_name,
            # 'Dominant': dominant
        }
        yaml_order = ['Pscav', 'Pcomp', 'Pmax', 'Texh', 'Ntc', 'Ntc_Pscav', 'Pcomp_Pscav', 'PR']
        # Map parameters from frontend names to internal names
        for internal_param in yaml_order:
            api_param = None
            for frontend_param, mapped_param in param_mapping.items():
                if mapped_param == internal_param:
                    api_param = frontend_param
                    break
            if api_param and api_param in fault_info:
                try:
                    value = int(fault_info[api_param])
                    if value not in [-1, 0, 1]:
                        raise ValueError(f"Parameter values must be -1, 0, or 1. Got {value} for {api_param}")
                    row[internal_param] = value
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid value for parameter '{api_param}': must be -1, 0, or 1")    
            else:
                row[internal_param] = 0
        row['Dominant'] = dominant        
        rows.append(row)
    
    return pd.DataFrame(rows)


def exists_consecutively(lst, element):
            count = 0
            for i in lst:
                if i == element:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0
            return False
def rating_level(row):
        rating = {}
        if (row['InjSysFault']>=0)&(row['InjSysFault']<30):                                   #60
            rating['InjSysFault'] = str(utility_dict['rating_level']['0-30'][1])     #[0-60]
        elif (row['InjSysFault']>=30)&(row['InjSysFault']<45):                                 #70
            rating['InjSysFault'] = str(utility_dict['rating_level']['30-45'][1])    #[60-70]
        elif (row['InjSysFault']>=45)&(row['InjSysFault']<65):                                  #80 
            rating['InjSysFault'] = str(utility_dict['rating_level']['45-65'][1])     #[70-80]
        elif row['InjSysFault']>=65:                                                             #80
            rating['InjSysFault'] = str(utility_dict['rating_level']['65-100'][1])  #[80-100]

        if (row['StaInjLate']>=0)&(row['StaInjLate']<30):
            rating['StaInjLate'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['StaInjLate']>=30)&(row['StaInjLate']<45):
            rating['StaInjLate'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['StaInjLate']>=45)&(row['StaInjLate']<65):
            rating['StaInjLate'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['StaInjLate']>=65:
            rating['StaInjLate'] = str(utility_dict['rating_level']['65-100'][1])   

        if (row['StaInjEarly']>=0)&(row['StaInjEarly']<30):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['StaInjEarly']>=30)&(row['StaInjEarly']<45):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['StaInjEarly']>=45)&(row['StaInjEarly']<65):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['StaInjEarly']>=65:
            rating['StaInjEarly'] = str(utility_dict['rating_level']['65-100'][1])    

        if (row['ExhValvLeak']>=0)&(row['ExhValvLeak']<30):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['ExhValvLeak']>=30)&(row['ExhValvLeak']<45):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['ExhValvLeak']>=45)&(row['ExhValvLeak']<65):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['ExhValvLeak']>=65:
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['65-100'][1])        

        if (row['BloCombChabr']>=0)&(row['BloCombChabr']<30):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['BloCombChabr']>=30)&(row['BloCombChabr']<45):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['BloCombChabr']>=45)&(row['BloCombChabr']<65):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['BloCombChabr']>=65:
            rating['BloCombChabr'] = str(utility_dict['rating_level']['65-100'][1])    

        if (row['ExhValEarOpn']>=0)&(row['ExhValEarOpn']<30):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['ExhValEarOpn']>=30)&(row['ExhValEarOpn']<45):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['30-45'][1])
        elif (row['ExhValEarOpn']>=45)&(row['ExhValEarOpn']<65):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['ExhValEarOpn']>=65:
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['65-100'][1])      

        if (row['ExhValLatOpn']>=0)&(row['ExhValLatOpn']<30):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['ExhValLatOpn']>=30)&(row['ExhValLatOpn']<45):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['ExhValLatOpn']>=45)&(row['ExhValLatOpn']<65):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['ExhValLatOpn']>=65:
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['65-100'][1])    

        if (row['ExhValEarlClos']>=0)&(row['ExhValEarlClos']<30):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['0-30'][1])
        elif (row['ExhValEarlClos']>=30)&(row['ExhValEarlClos']<45):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['ExhValEarlClos']>=45)&(row['ExhValEarlClos']<65):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['ExhValEarlClos']>=65:
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['65-100'][1])      

        if (row['ExhValLatClos']>=0)&(row['ExhValLatClos']<30):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['0-30'][1]) 
        elif (row['ExhValLatClos']>=30)&(row['ExhValLatClos']<45):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['30-45'][1]) 
        elif (row['ExhValLatClos']>=45)&(row['ExhValLatClos']<65):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['45-65'][1])     
        elif row['ExhValLatClos']>=65:
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['65-100'][1])         
        return rating
def output_format(rl,typo,des,utility_dict,Ftype):
    # rl - 
        if utility_dict!=None:
            er=','.join(utility_dict['faults_recom'][Ftype][des])
            return '({})({})-{}-{}'.format(str(rl),typo,'There is a high chance for the occurrence of this fault within two weeks;'+des,er)
        else:
            er='All values are fine for this fault'
            return '({})({})-{}-{}'.format(str(rl),typo,des,er)


# df = pd.read_csv(r'C:\Users\User\Desktop\FaultSenseAI_source\smartmaintanace\mgd.csv')
def run_pipeline(data_path, config_overrides=None):
    utility_dict = load_config_with_overrides(config_overrides)
    
    df = pd.read_csv(data_path)
    df['signaldate'] = pd.to_datetime(df['signaldate'], dayfirst=True)
    df.set_index('signaldate', inplace=True)
    df_hourly = df.resample('1H').mean()

    input_data = df_hourly.iloc[-43*24:-29*24].copy()

    # All your existing preprocessing (keep everything the same)
    column_mapping = {}
    for required_col in utility_dict['imp_feats_ts']:
        search_col = required_col.replace('(ems)', '').replace('(ams)', '').replace('(scr)', '')
        for csv_col in input_data.columns:
            if csv_col.strip() == search_col.strip():
                column_mapping[csv_col] = required_col
                break

    input_data = input_data.rename(columns=column_mapping)

    # Create derived features
    input_data['PR'] = input_data['Firing Pressure Average(ems)'] - input_data['Compression Pressure Average(ems)']
    input_data['Ntc_Pscav'] = input_data['Turbocharger 1 speed(ems)'] / input_data['Scav. Air Press. Mean Value(ems)']
    input_data['Pcomp_Pscav'] = input_data['Compression Pressure Average(ems)'] / input_data['Scav. Air Press. Mean Value(ems)']

    # Calculate cylinder features (your existing code)
    exh_valve_cols = [f'Exh. valve opening angle Cyl #{i+1:02d}' for i in range(6)]
    available_exh_cols = [col for col in exh_valve_cols if col in input_data.columns]
    if available_exh_cols:
        input_data['Exh. valve opening angle Cyl_mean'] = input_data[available_exh_cols].mean(axis=1)
        input_data['Exh. valve opening angle Cyl_min'] = input_data[available_exh_cols].min(axis=1)
        input_data['Exh. valve opening angle Cyl_max'] = input_data[available_exh_cols].max(axis=1)

    firing_cols = [f'Firing Pr. Balancing Injection Offset Cyl #{i+1:02d}' for i in range(6)]
    available_firing_cols = [col for col in firing_cols if col in input_data.columns]
    if available_firing_cols:
        input_data['Firing Pr. Balancing Injection Offset Cyl_mean'] = input_data[available_firing_cols].mean(axis=1)
        input_data['Firing Pr. Balancing Injection Offset Cyl_min'] = input_data[available_firing_cols].min(axis=1)
        input_data['Firing Pr. Balancing Injection Offset Cyl_max'] = input_data[available_firing_cols].max(axis=1)

    start_inj_cols = [f'Start of Injection Cyl #{i+1:02d}' for i in range(6)]
    available_start_cols = [col for col in start_inj_cols if col in input_data.columns]
    if available_start_cols:
        input_data['Start of Injection Cyl_mean'] = input_data[available_start_cols].mean(axis=1)
        input_data['Start of Injection Cyl_min'] = input_data[available_start_cols].min(axis=1)
        input_data['Start of Injection Cyl_max'] = input_data[available_start_cols].max(axis=1)

    # NEW: Use all 336 hours without engine load filter
    model_input_columns = utility_dict['imp_feats_ts']
    model_ready_data = input_data[model_input_columns]

    # print(f"Final model input shape: {model_ready_data.shape}")
    # print("Data is ready for the model!")


    # Add this to your existing code:

    # Load the scalers and model
    scaler_x = load(utility_dict['TS_scale_x'])  # TS_X_scaler.joblib
    scaler_y = load(utility_dict['TS_scale_y'])  # TS_Y_scaler.joblib

    # print("Scalers loaded successfully")

    # Scale the input data
    scaled_input = scaler_x.transform(model_ready_data)
    # print(f"Scaled input shape: {scaled_input.shape}")

    # Add positional encoding (like your original TS model does)
    positions = np.array(range(1, 337)) / 336  # 1 to 336, normalized
    scaled_input_with_pos = np.append(scaled_input, positions.reshape(-1, 1), axis=1)
    # print(f"Input with positions shape: {scaled_input_with_pos.shape}")

    # Reshape for model input (batch_size=1, sequence_length=336, features=41)
    X_model_input = scaled_input_with_pos.reshape((1, 336, 41))
    # print(f"Final model input shape: {X_model_input.shape}")

    # Load and use the model
    model = tf.keras.models.load_model(utility_dict['TS_model_path'])
    # print("Model loaded successfully")

    # Make prediction
    # print("Making prediction...")
    predictions = model.predict([X_model_input, tf.zeros((1, 336, 9))])
    # print(f"Raw predictions shape: {predictions.shape}")

    # Inverse transform predictions
    predictions_real = scaler_y.inverse_transform(predictions[0])
    # print(f"Final predictions shape: {predictions_real.shape}")

    print("Prediction complete!")

    # Add this to see your predictions:

    # Convert predictions to DataFrame with proper column names
    prediction_columns = utility_dict['TS_frame_colnames']  # 9 output features
    predictions_df = pd.DataFrame(predictions_real, columns=prediction_columns)

    
    actual_comparison = df_hourly.iloc[-29*24:-15*24]  # Days -29 to -15

    # Apply the same preprocessing to actual data
    actual_processed = actual_comparison.rename(columns=column_mapping)
    actual_processed['PR'] = actual_processed['Firing Pressure Average(ems)'] - actual_processed['Compression Pressure Average(ems)']
    actual_processed['Ntc_Pscav'] = actual_processed['Turbocharger 1 speed(ems)'] / actual_processed['Scav. Air Press. Mean Value(ems)']
    actual_processed['Pcomp_Pscav'] = actual_processed['Compression Pressure Average(ems)'] / actual_processed['Scav. Air Press. Mean Value(ems)']


    # Create timestamps for predictions (starting from where input data ended)
    last_input_time = input_data.index[-1]
    prediction_times = pd.date_range(start=last_input_time + pd.Timedelta(hours=1), 
                                    periods=336, freq='H')
    predictions_df.index = prediction_times

    print(f"\nPrediction time range: {predictions_df.index[0]} to {predictions_df.index[-1]}")
    print("Success! You have 14-day predictions ready for analysis.")


    
    reshaped_predictions = predictions[0].reshape(-1, predictions.shape[-1])
    

    try:
        manual_inverse = scaler_y.inverse_transform(reshaped_predictions)
        print(f"Manual inverse successful: {manual_inverse.shape}")
        print(f"Sample values: {manual_inverse[0]}")
    except Exception as e:
        print(f"Manual inverse failed: {e}")


    
    nan_columns = model_ready_data.columns[model_ready_data.isnull().any()]
    print(list(nan_columns))

    # If there are NaN values, let's see which ones
    if len(nan_columns) > 0:
        # print(f"\nNaN counts per column:")
        for col in nan_columns:
            nan_count = model_ready_data[col].isnull().sum()
            print(f"{col}: {nan_count} NaN values")


    

    # Option 1: Forward fill (use last known value)
    model_ready_data_fixed = model_ready_data.fillna(method='ffill')

    # Option 2: If still NaN at the beginning, backward fill
    model_ready_data_fixed = model_ready_data_fixed.fillna(method='bfill')

    # Option 3: If still any NaN, fill with column mean
    model_ready_data_fixed = model_ready_data_fixed.fillna(model_ready_data_fixed.mean())

    # Now rescale and predict with clean data
    scaled_input_fixed = scaler_x.transform(model_ready_data_fixed)
    positions = np.array(range(1, 337)) / 336
    scaled_input_with_pos_fixed = np.append(scaled_input_fixed, positions.reshape(-1, 1), axis=1)
    X_model_input_fixed = scaled_input_with_pos_fixed.reshape((1, 336, 41))

    # print("Making prediction with fixed data...")
    predictions_fixed = model.predict([X_model_input_fixed, tf.zeros((1, 336, 9))])


    if not np.isnan(predictions_fixed).any():
        predictions_real_fixed = scaler_y.inverse_transform(predictions_fixed[0])
        predictions_df_fixed = pd.DataFrame(predictions_real_fixed, columns=prediction_columns)
        
    else:
        print("Still getting NaN predictions - model or scaler issue")



    print('###############################################################')
    # Now let's compare with what actually happened
    # Get the actual data for the same period (days -29 to -15)
    actual_comparison = df_hourly.iloc[-29*24:-15*24].copy()

    # Apply same preprocessing to actual data
    actual_comparison = actual_comparison.rename(columns=column_mapping)

    # Get the same columns that we predicted
    actual_features = ['Compression Pressure Average(ems)', 'Scav. Air Press. Mean Value(ems)', 
                    'Exhaust Gas Average Temperature(ems)', 'Turbocharger 1 speed(ems)',
                    'Firing Pressure Average(ems)', 'Estimated engine load(ems)']

   
    print("-" * 55)

    for i, col in enumerate(['TS_Pcomp', 'TS_Pscav', 'TS_Texh', 'TS_Ntc', 'TS_Pmax', 'Estimated engine load(ems)']):
        if col in predictions_df_fixed.columns and actual_features[i] in actual_comparison.columns:
            pred_mean = predictions_df_fixed[col].mean()
            actual_mean = actual_comparison[actual_features[i]].mean()
            diff = abs(pred_mean - actual_mean)
            print(f"{col[:15]:15} | {pred_mean:12.2f} | {actual_mean:11.2f} | {diff:10.2f}")

    # Add timestamps and save predictions
    last_input_time = input_data.index[-1]
    prediction_times = pd.date_range(start=last_input_time + pd.Timedelta(hours=1), 
                                    periods=336, freq='H')
    predictions_df_fixed.index = prediction_times

    # Reset index to make datetime a column
    predictions_csv = predictions_df_fixed.reset_index()
    predictions_csv.rename(columns={'index': 'datetime'}, inplace=True)

    # Save to CSV
    predictions_csv.to_csv('ts_predictions.csv', index=False)
    print("Predictions saved to ts_predictions.csv")
    print(predictions_csv.head())   

    ####### AUTOENCODER ####

    print("\n=== PREPARING AUTOENCODER DATA ===")

    # Autoencoder needs much more historical data
    historical_data = df_hourly.iloc[:-14*24].copy()  # Everything except last 14 days
    recent_data = df_hourly.iloc[-14*24:].copy()     # Last 14 days

    # Combine them (like the original code does)
    ae_full_data = pd.concat([historical_data, recent_data])

    # Apply same preprocessing as TS model
    ae_full_data = ae_full_data.rename(columns=column_mapping)

    # Filter engine load (30-100%) like the original code
    ae_full_data = ae_full_data[(ae_full_data['Estimated engine load(ems)'] >= 30) & 
                                (ae_full_data['Estimated engine load(ems)'] <= 100)]

    # Add derived features
    ae_full_data['Ntc_Pscav'] = ae_full_data['Turbocharger 1 speed(ems)'] / ae_full_data['Scav. Air Press. Mean Value(ems)']
    ae_full_data['PR'] = ae_full_data['Firing Pressure Average(ems)'] - ae_full_data['Compression Pressure Average(ems)']
    ae_full_data['Pcomp_Pscav'] = ae_full_data['Compression Pressure Average(ems)'] / ae_full_data['Scav. Air Press. Mean Value(ems)']

    # Handle NaN values
    ae_full_data = ae_full_data.fillna(method='ffill').fillna(method='bfill').fillna(ae_full_data.mean())

    # Save autoencoder input
    ae_full_data.to_csv('ae_input_data.csv')
    print("Autoencoder input saved to ae_input_data.csv")


    # Add this to load and run the autoencoder:

    print("\n=== LOADING AUTOENCODER MODEL ===")

    # Load autoencoder model and scaler
    from keras.models import load_model
    # from tensorflow.keras.models import load_model

    ae_model_path = utility_dict['vessel_spec']['Mu Lan']['Ae_model_path']
    ae_scaler_path = utility_dict['vessel_spec']['Mu Lan']['Ae_scalar']

    ae_scaler = load(ae_scaler_path)
    ae_model = load_model(ae_model_path)

    # print("Autoencoder model and scaler loaded successfully")

    # Prepare data for autoencoder (it needs specific columns)
    # The AE was trained on specific features - let's use the main engine parameters
    ae_features = ['Compression Pressure Average(ems)', 'Scav. Air Press. Mean Value(ems)', 
                'Exhaust Gas Average Temperature(ems)', 'Turbocharger 1 speed(ems)',
                'Firing Pressure Average(ems)', 'Estimated engine load(ems)',
                'PR', 'Ntc_Pscav', 'Pcomp_Pscav']

    print("Features the AE scaler expects:")
    expected_features = ae_scaler.feature_names_in_
    print(f"Total expected features: {len(expected_features)}")
    print("First 10 expected features:", expected_features[:10])

    # Check which ones we have in our data
    available_features = []
    missing_features = []

    for feature in expected_features:
        if feature in ae_full_data.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)

    # print(f"\nWe have {len(available_features)} out of {len(expected_features)} features")
    # print(f"Missing {len(missing_features)} features")

    # Create a dataframe with all expected features
    ae_input_complete = pd.DataFrame(index=ae_full_data.index)

    # Add available features
    for feature in available_features:
        ae_input_complete[feature] = ae_full_data[feature]

    # Fill missing features with zeros or mean values
    for feature in missing_features:
        ae_input_complete[feature] = 0  # or use mean from training data if available

    # Fill NaN values
    ae_input_complete = ae_input_complete.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Ensure columns are in the same order as scaler expects
    ae_input_ordered = ae_input_complete[expected_features]

    # print(f"Final AE input shape: {ae_input_ordered.shape}")

    # Now scale and run autoencoder
    ae_scaled = ae_scaler.transform(ae_input_ordered)
    ae_reconstructed = ae_model.predict(ae_scaled)

    # Calculate reconstruction error
    mse_loss = np.mean(np.square(ae_scaled - ae_reconstructed), axis=1)

    print(f"Autoencoder completed successfully!")
    print(f"Mean reconstruction error: {mse_loss.mean():.4f}")


    # Save autoencoder anomaly detection results
    ae_results = pd.DataFrame({
        'datetime': ae_full_data.index,
        'reconstruction_error': mse_loss
    })

    # Add anomaly flag (threshold: mean + 2*std)
    threshold = mse_loss.mean() + 2 * mse_loss.std()
    ae_results['is_anomaly'] = ae_results['reconstruction_error'] > threshold

    ae_results.to_csv('autoencoder_anomaly_results.csv', index=False)
    print(f"Autoencoder results saved to autoencoder_anomaly_results.csv")
    # print(f"Anomalies detected: {ae_results['is_anomaly'].sum()} out of {len(ae_results)} samples")

    # Filter out anomalies before ML processing
    print(f"Filtering out {ae_results['is_anomaly'].sum()} anomalous samples")
    normal_mask = ~ae_results['is_anomaly']  # NOT anomaly
    normal_indices = ae_results[normal_mask].index

    # Use only normal operational data for ML models
    ml_input_data = ae_full_data.loc[ae_full_data.index.isin(normal_indices)].copy()
    print(f"ML input data after anomaly filtering: {ml_input_data.shape}")

    # Set engine number (you can use 1 or 2)
    eng = 1  

    print(f"\n=== PREPARING FOR ML MODELS ===")
    print(f"ML input shape: {ml_input_data.shape}")
    print("Data ready for ML.ML_models()")


    # From your YAML, you have 8 EFD features and 6 cylinders
    efd_features = utility_dict['efd_features']  # ['Pcomp_Pscav','PR','Pmax','Ntc','Ntc_Pscav','Pcomp','Pscav','Texh']
    cyl_count = utility_dict['cyl_count']  # 6

    print("Next: ML models will predict EFD features for each cylinder")
    print(f"EFD features to predict: {efd_features}")
    print(f"For {cyl_count} cylinders")


    # Create directories for results
    os.makedirs('./TS_results/', exist_ok=True)
    os.makedirs('./ML_results/', exist_ok=True)

    # First, format TS predictions for ML model consumption
    print("\n=== FORMATTING TS PREDICTIONS FOR ML MODELS ===")

    # Load our TS predictions
    ts_preds = pd.read_csv('ts_predictions.csv') if os.path.exists('ts_predictions.csv') else predictions_df_fixed.reset_index()
    if 'datetime' not in ts_preds.columns and 'index' in ts_preds.columns:
        ts_preds.rename(columns={'index': 'datetime'}, inplace=True)

    ts_preds['datetime'] = pd.to_datetime(ts_preds['datetime'])
    ts_preds.set_index('datetime', inplace=True)

    # Create TS result files for each cylinder
    eng = 1
    today_str = datetime.now().strftime("%Y-%m-%d")

    for cyl in range(1, cyl_count + 1):
        ts_cyl = ts_preds.copy()
        ts_cyl.reset_index(inplace=True)
        ts_cyl.rename(columns={'datetime': 'Date Time'}, inplace=True)
        
        # Add required columns that ML model expects
        ts_cyl['Estimated engine load(ems)'] = ts_cyl['Estimated engine load(ems)'] if 'Estimated engine load(ems)' in ts_cyl.columns else ts_preds['Estimated engine load(ems)'].values
        
        filename = f'Mu Lan_ENG_{eng}_TS_res_Cyl_{cyl}_{today_str}.csv'
        ts_cyl.to_csv(f'./TS_results/{filename}', index=False)

    print("TS result files created for each cylinder")

    # Create modified ML model class for local operation
    class pdm_ml_model_local:
        def __init__(self, Efd_features, ts_res, engine_number, ml_res, vessel_name):
            self.Efd_features = Efd_features
            self.ts_res = ts_res
            self.engine_number = engine_number
            self.ml_res = ml_res
            self.cyl_count = utility_dict['vessel_spec'][vessel_name]['cyl_no']
            self.load_limit = utility_dict['load_limit']
            self.max_load = utility_dict['max_load']
            
            # Load all the ML models and scalers
            print("Loading ML models and scalers...")
            self.Pcomp_Pscav_scaler_x = load(utility_dict['Pcomp_Pscav_scaler_x']) 
            self.Pcomp_Pscav_scaler_y = load(utility_dict['Pcomp_Pscav_scaler_y']) 
            self.Pcomp_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_Pscav_ml_model'])
            
            self.PR_scaler_x = load(utility_dict['PR_scaler_x'])
            self.PR_scaler_y = load(utility_dict['PR_scaler_y']) 
            self.PR_ml_model = tf.keras.models.load_model(utility_dict['PR_ml_model'])
            
            self.Ntc_Pscav_scaler_x = load(utility_dict['Ntc_Pscav_scaler_x'])
            self.Ntc_Pscav_scaler_y = load(utility_dict['Ntc_Pscav_scaler_y']) 
            self.Ntc_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Ntc_Pscav_ml_model'])
            
            self.Pmax_scaler_x = load(utility_dict['Pmax_scaler_x'])
            self.Pmax_scaler_y = load(utility_dict['Pmax_scaler_y'])
            self.Pmax_ml_model = tf.keras.models.load_model(utility_dict['Pmax_ml_model'])
            
            self.Texh_scaler_x = load(utility_dict['Texh_scaler_x'])
            self.Texh_scaler_y = load(utility_dict['Texh_scaler_y'])
            self.Texh_ml_model = tf.keras.models.load_model(utility_dict['Texh_ml_model'])
            
            self.Ntc_scaler_x = load(utility_dict['Ntc_scaler_x'])
            self.Ntc_scaler_y = load(utility_dict['Ntc_scaler_y']) 
            self.Ntc_ml_model = tf.keras.models.load_model(utility_dict['Ntc_ml_model'])
            
            self.Pcomp_scaler_x = load(utility_dict['Pcomp_scaler_x'])
            self.Pcomp_scaler_y = load(utility_dict['Pcomp_scaler_y'])
            self.Pcomp_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_ml_model'])
            
            self.Pscav_scaler_x = load(utility_dict['Pscav_scaler_x'])
            self.Pscav_scaler_y = load(utility_dict['Pscav_scaler_y'])
            self.Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pscav_ml_model'])
            
            print("All ML models and scalers loaded successfully")
        
        def ML_models(self, data, eng, vessel_name):
            print(f"Running ML models for Engine {eng}")
            
            # Filter data like original
            df2 = data.copy()
            df2 = df2[(df2[utility_dict['engine_load']] >= 30) & (df2[utility_dict['engine_load']] <= 100)]
            df2 = df2.fillna(method='ffill').fillna(method='bfill').fillna(df2.mean())
            
            for cyl in range(1, self.cyl_count + 1):
                print(f"Processing Cylinder {cyl}...")
                
                # Read TS results for this cylinder
                filename = f'{vessel_name}_ENG_{eng}_TS_res_Cyl_{cyl}_{today_str}.csv'
                filepath = os.path.join(self.ts_res, filename)
                cyl_df = pd.read_csv(filepath, index_col=False)
                
                # Process load matching (simplified version)
                load_ranges = cyl_df[utility_dict['engine_load']].unique()
                
                for loads in load_ranges:
                    load_l_limit = loads * ((100 - self.load_limit) / 100)
                    load_u_limit = loads * ((100 + self.load_limit) / 100)
                    
                    load_cons = df2[(df2[utility_dict['engine_load']] >= load_l_limit) & 
                                (df2[utility_dict['engine_load']] <= load_u_limit)]
                    
                    if len(load_cons) > 0:
                        # Find best match
                        load_cons = load_cons.copy()
                        load_cons['load_delta'] = abs(load_cons[utility_dict['engine_load']] - loads)
                        best_match = load_cons.loc[load_cons['load_delta'].idxmin()]
                        
                        # Update cyl_df
                        mask = cyl_df[utility_dict['engine_load']] == loads
                        cyl_df.loc[mask, 'matched_load'] = best_match[utility_dict['engine_load']]
                        cyl_df.loc[mask, 'matched_date'] = best_match.name
                        cyl_df.loc[mask, 'deltas'] = best_match['load_delta']
                    else:
                        # No match found
                        mask = cyl_df[utility_dict['engine_load']] == loads
                        cyl_df.loc[mask, 'matched_load'] = 'No Values'
                        cyl_df.loc[mask, 'matched_date'] = 'No Values'
                        cyl_df.loc[mask, 'deltas'] = 'No Values'
                
                # Get valid indices (not 'No Values')
                valid_mask = cyl_df['matched_load'] != 'No Values'
                valid_indices = cyl_df[valid_mask].index
                
                if len(valid_indices) == 0:
                    print(f"No valid matches for Cylinder {cyl}, skipping...")
                    continue
                
                # Prepare data for ML models
                matched_dates = cyl_df.loc[valid_indices, 'matched_date'].values
                df_for_ml = pd.DataFrame()
                
                for date_idx in matched_dates:
                    if date_idx in df2.index:
                        df_for_ml = pd.concat([df_for_ml, df2.loc[[date_idx]]])
                
                if df_for_ml.empty:
                    print(f"No data found for Cylinder {cyl}, skipping...")
                    continue
                    
                # Run ML predictions for each EFD feature
                for efd_idx, efds in enumerate(self.Efd_features):
                    try:
                        if efds == utility_dict['efd_features'][0]:  # Pcomp_Pscav
                            model_inputs = df_for_ml[self.Pcomp_Pscav_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Pcomp_Pscav_scaler_x.transform(model_inputs),
                                columns=self.Pcomp_Pscav_scaler_x.feature_names_in_
                            )
                            y_pred = self.Pcomp_Pscav_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Pcomp_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][1]:  # PR
                            model_inputs = df_for_ml[self.PR_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.PR_scaler_x.transform(model_inputs),
                                columns=self.PR_scaler_x.feature_names_in_
                            )
                            y_pred = self.PR_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.PR_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][4]:  # Ntc_Pscav
                            model_inputs = df_for_ml[self.Ntc_Pscav_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Ntc_Pscav_scaler_x.transform(model_inputs),
                                columns=self.Ntc_Pscav_scaler_x.feature_names_in_
                            )
                            y_pred = self.Ntc_Pscav_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Ntc_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][2]:  # Pmax
                            model_inputs = df_for_ml[self.Pmax_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Pmax_scaler_x.transform(model_inputs),
                                columns=self.Pmax_scaler_x.feature_names_in_
                            )
                            y_pred = self.Pmax_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Pmax_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][7]:  # Texh
                            model_inputs = df_for_ml[self.Texh_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Texh_scaler_x.transform(model_inputs),
                                columns=self.Texh_scaler_x.feature_names_in_
                            )
                            y_pred = self.Texh_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Texh_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][3]:  # Ntc
                            model_inputs = df_for_ml[self.Ntc_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Ntc_scaler_x.transform(model_inputs),
                                columns=self.Ntc_scaler_x.feature_names_in_
                            )
                            y_pred = self.Ntc_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Ntc_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][5]:  # Pcomp
                            model_inputs = df_for_ml[self.Pcomp_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Pcomp_scaler_x.transform(model_inputs),
                                columns=self.Pcomp_scaler_x.feature_names_in_
                            )
                            y_pred = self.Pcomp_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Pcomp_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                            
                        elif efds == utility_dict['efd_features'][6]:  # Pscav
                            model_inputs = df_for_ml[self.Pscav_scaler_x.feature_names_in_]
                            model_inputs_scaled = pd.DataFrame(
                                self.Pscav_scaler_x.transform(model_inputs),
                                columns=self.Pscav_scaler_x.feature_names_in_
                            )
                            y_pred = self.Pscav_ml_model.predict(model_inputs_scaled, verbose=0)
                            y_pred = self.Pscav_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
                        
                        # Store predictions
                        cyl_df.loc[valid_indices, f'Ref_{efds}'] = [pred[0] for pred in y_pred]
                        
                    except Exception as e:
                        print(f"Error processing {efds} for cylinder {cyl}: {e}")
                        cyl_df.loc[valid_indices, f'Ref_{efds}'] = 'Error'
                
                # Save results
                output_filename = f'{vessel_name}_ENG_{eng}_TS_ML_res_Cyl_{cyl}_{today_str}.csv'
                output_filepath = os.path.join(self.ml_res, output_filename)
                cyl_df.to_csv(output_filepath, index=False)
                
            print("ML models completed for all cylinders!")



    print("\n=== FIXING ML INPUT DATA ===")

    # The ML models need the cylinder-based features we calculated earlier
    # Let's recreate ml_input_data with ALL the features, including the calculated ones
    ml_input_data_fixed = ae_full_data.copy()

    # Ensure all the cylinder-based calculated features are present
    # (These were calculated earlier for TS model but ML models also need them)

    # Add any missing calculated features if needed
    exh_valve_cols = [col for col in ml_input_data_fixed.columns if 'Exh. valve opening angle Cyl #' in col]
    if len(exh_valve_cols) >= 3 and 'Exh. valve opening angle Cyl_mean' not in ml_input_data_fixed.columns:
        ml_input_data_fixed['Exh. valve opening angle Cyl_mean'] = ml_input_data_fixed[exh_valve_cols].mean(axis=1)
        ml_input_data_fixed['Exh. valve opening angle Cyl_min'] = ml_input_data_fixed[exh_valve_cols].min(axis=1)
        ml_input_data_fixed['Exh. valve opening angle Cyl_max'] = ml_input_data_fixed[exh_valve_cols].max(axis=1)

    firing_cols = [col for col in ml_input_data_fixed.columns if 'Firing Pr. Balancing Injection Offset Cyl #' in col]
    if len(firing_cols) >= 3 and 'Firing Pr. Balancing Injection Offset Cyl_mean' not in ml_input_data_fixed.columns:
        ml_input_data_fixed['Firing Pr. Balancing Injection Offset Cyl_mean'] = ml_input_data_fixed[firing_cols].mean(axis=1)
        ml_input_data_fixed['Firing Pr. Balancing Injection Offset Cyl_min'] = ml_input_data_fixed[firing_cols].min(axis=1)
        ml_input_data_fixed['Firing Pr. Balancing Injection Offset Cyl_max'] = ml_input_data_fixed[firing_cols].max(axis=1)

    start_inj_cols = [col for col in ml_input_data_fixed.columns if 'Start of Injection Cyl #' in col]
    if len(start_inj_cols) >= 3 and 'Start of Injection Cyl_mean' not in ml_input_data_fixed.columns:
        ml_input_data_fixed['Start of Injection Cyl_mean'] = ml_input_data_fixed[start_inj_cols].mean(axis=1)
        ml_input_data_fixed['Start of Injection Cyl_min'] = ml_input_data_fixed[start_inj_cols].min(axis=1)
        ml_input_data_fixed['Start of Injection Cyl_max'] = ml_input_data_fixed[start_inj_cols].max(axis=1)

    # Fill any remaining NaN values
    ml_input_data_fixed = ml_input_data_fixed.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Check if we have the required features now
    sample_scaler = load(utility_dict['Pcomp_Pscav_scaler_x'])
    required_features = sample_scaler.feature_names_in_
    missing_features = [f for f in required_features if f not in ml_input_data_fixed.columns]
    print(f"Still missing {len(missing_features)} features: {missing_features[:5]}...")

    # Replace ml_input_data with the fixed version
    ml_input_data = ml_input_data_fixed


    # Initialize and run ML models
    print("\n=== RUNNING ML MODELS ===")

    Efd_features = utility_dict['efd_features']
    ts_res_loc = './TS_results/'
    engine_number = utility_dict['engine_number']
    ml_res_loc = './ML_results/'

    ML_local = pdm_ml_model_local(Efd_features, ts_res_loc, engine_number, ml_res_loc, 'Mu Lan')

    # Run ML models
    # ML_result = ML_local.ML_models(ml_input_data, eng, 'Mu Lan')
    ML_result = ML_local.ML_models(ml_input_data_fixed, eng, 'Mu Lan')

    print("\n=== ML MODELS COMPLETE ===")
    print("Results saved in ./ML_results/ directory")
    print("Next step: Fault mapping to convert EFD predictions to fault probabilities")


    print("\n=== RUNNING FAULT MAPPING ===")

    from Fault_Mapping import Faults_Mapping
    

    # Parameters for fault mapping
    fault_mat_loc = utility_dict['f_mat_path']  # Fault matrix Excel file path
    p_value = utility_dict['p_value']  # Weight value for KPI calculations
    Efd_features = utility_dict['efd_features']

    # Process each cylinder's ML results
    final_results = {}
    eng = 1

    for cyl in range(1, cyl_count + 1):
        # print(f"Processing fault mapping for Cylinder {cyl}...")
        
        # Read ML results for this cylinder
        ml_result_file = f'Mu Lan_ENG_{eng}_TS_ML_res_Cyl_{cyl}_{today_str}.csv'
        ml_result_path = os.path.join('./ML_results/', ml_result_file)
        
        if os.path.exists(ml_result_path):
            ml_ress = pd.read_csv(ml_result_path, index_col=False)
        
            for col in ml_ress.columns:
                if 'Ref_' in col:
                    unique_vals = ml_ress[col].unique()
                    # print(f"{col} unique values: {unique_vals}")
                    non_numeric = ml_ress[col].apply(lambda x: not str(x).replace('.', '').replace('-', '').replace('nan', '').isdigit() if str(x) != 'nan' else False)
                    if non_numeric.any():
                        print(f"NON-NUMERIC VALUES FOUND IN {col}: {ml_ress[col][non_numeric].values}")
    
    # Check fault matrix Excel
            fault_matrix_df = pd.read_excel(fault_mat_loc)
            fault_mapper = Faults_Mapping(ml_ress, fault_mat_loc, Efd_features, p_value)
            # print("Calling Mapping() method...")
            fault_results, fault_ids = fault_mapper.Mapping()
            ml_ress_with_faults = pd.concat([ml_ress, fault_results[fault_ids]], axis=1)
            
            
            # Create timestamps for the forecast period (next 14 days)
            last_input_time = input_data.index[-1]
            forecast_times = pd.date_range(start=last_input_time + pd.Timedelta(hours=1), 
                                        periods=len(ml_ress_with_faults), freq='H')
            
            ml_ress_with_faults['Date Time'] = forecast_times
            
            # Reorder columns to match expected format
            if 'end_res_colorder' in utility_dict:
                available_cols = [col for col in utility_dict['end_res_colorder'] if col in ml_ress_with_faults.columns]
                ml_ress_with_faults = ml_ress_with_faults[available_cols]
            
            # Save final results
            final_result_file = f'Mu Lan_Eng_{eng}_mapping_res_cyl{cyl}_{today_str}.csv'
            final_result_path = os.path.join('./Final_results/', final_result_file)
            os.makedirs('./Final_results/', exist_ok=True)
            
            ml_ress_with_faults.to_csv(final_result_path, index=False)
            
            # Store in dictionary for final output
            final_results[f'Cyl_{cyl}'] = ml_ress_with_faults.to_dict(orient='list')
            
            print(f"Cylinder {cyl} fault mapping completed")
        else:
            print(f"ML results file not found for Cylinder {cyl}")

    print("\n=== GENERATING FINAL MAINTENANCE RECOMMENDATIONS ===")

    # Create final output format (similar to maindup.py output)
    output_format_mapping = {
        'Vessel_info': {
            'Vessel_Name': 'Mu Lan',
            'VESSEL_OBJECT_ID': utility_dict['VESSEL_OBJECT_ID'],
            'JOB_PLAN_ID': utility_dict['JOB_PLAN_ID']
        },
        'Engine_data': {
            'Engine_1': final_results
        }
    }

    # Save complete results
    # import pickle
    with open(f'complete_maintenance_results_{today_str}.pickle', 'wb') as f:
        pickle.dump(output_format_mapping, f)

    
    
    return {
        "status": "success", 
        "message": "Pipeline completed",
        "predictions_shape": str(predictions_df_fixed.shape) if 'predictions_df_fixed' in locals() else "unknown"
    }


def process_smart_maintenance_results(pickle_file_path, utility_dict):
    print("=== INSIDE process_smart_maintenance_results ===")
    try:
        with open(pickle_file_path, 'rb') as f:
            end_point_result = pickle.load(f)
        print("Pickle file loaded successfully")    
        
        api_format = {'totalRecords': 0, 'surveys': []}
        print("API format initialized")

        
        # Get measured_date from Date Time column
        try:
            cyl_1_data = end_point_result['Engine_data']['Engine_1']['Cyl_1']
            print(f"Cylinder 1 data found: {len(cyl_1_data) if cyl_1_data else 0} keys")
            if 'Date Time' in cyl_1_data and len(cyl_1_data['Date Time']) > 0:
                measured_date = str(cyl_1_data['Date Time'][0]).split()[0]
            else:
                measured_date = datetime.now().strftime("%Y-%m-%d")
        except:
            print(f"Error getting date")
            measured_date = datetime.now().strftime("%Y-%m-%d")
        
        upload_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load Excel files
        try:
            me1_faults = pd.read_excel('MuLAN Details.xlsx', sheet_name='ME1_filt')
            me2_faults = pd.read_excel('MuLAN Details.xlsx', sheet_name='ME2_filt')
            print("Excel files loaded successfully")
        except:
            return {"totalRecords": 0, "surveys": []}
        
        # Process fault categories
        for fault_cats in utility_dict['Fault_cats_ids'].keys():
            print(f"Processing fault category: {fault_cats}") 
            for engs in end_point_result['Engine_data'].keys():
                for engs_cyl in end_point_result['Engine_data'][engs].keys():
                    try:
                        cyl_data = end_point_result['Engine_data'][engs][engs_cyl]
                        
                        # Convert to DataFrame
                        cyl_df = pd.DataFrame(cyl_data)
                        
                        if cyl_df.empty or 'Date Time' not in cyl_df.columns:
                            continue
                        
                        # Check if required fault columns exist
                        required_cols = ['InjSysFault', 'StaInjLate', 'StaInjEarly', 'ExhValvLeak', 
                                       'BloCombChabr', 'ExhValEarOpn', 'ExhValLatOpn', 'ExhValEarlClos', 'ExhValLatClos']
                        print(f"Available columns in cylinder data: {cyl_df.columns.tolist()}")
                        missing_cols = [col for col in required_cols if col not in cyl_df.columns]
                        if missing_cols:
                            print(f"Missing required columns: {missing_cols}") 
                        if not all(col in cyl_df.columns for col in required_cols):
                            continue
                        
                        # Convert numeric fault data to finding strings using main.py format
                        findings = []
                        for idx, row in cyl_df.iterrows():
                            ratings = rating_level(row)
                            finding_parts = []
                            
                            # Map to exact strings from main.py
                            if 'InjSysFault' in ratings:
                                finding_parts.append(f"Injection system fault({ratings['InjSysFault']})")
                            if 'StaInjLate' in ratings:
                                finding_parts.append(f"Start of injection late({ratings['StaInjLate']})")
                            if 'StaInjEarly' in ratings:
                                finding_parts.append(f"Start of injection early({ratings['StaInjEarly']})")
                            if 'ExhValvLeak' in ratings:
                                finding_parts.append(f"Exhaust valve leak({ratings['ExhValvLeak']})")
                            if 'BloCombChabr' in ratings:
                                finding_parts.append(f"Blow-by in combustion chamber({ratings['BloCombChabr']})")
                            if 'ExhValEarOpn' in ratings:
                                finding_parts.append(f"Exhaust valve early opening({ratings['ExhValEarOpn']})")
                            if 'ExhValLatOpn' in ratings:
                                finding_parts.append(f"Exhaust valve late opening({ratings['ExhValLatOpn']})")
                            if 'ExhValEarlClos' in ratings:
                                finding_parts.append(f"Exhaust valve early closing({ratings['ExhValEarlClos']})")
                            if 'ExhValLatClos' in ratings:
                                finding_parts.append(f"Exhaust valve late closing({ratings['ExhValLatClos']})")
                            
                            findings.append('||'.join(finding_parts))
                        
                        # Create DataFrame with findings using Date Time as index (main.py format)
                        f1 = pd.DataFrame({'finding': findings}, index=cyl_df['Date Time'])
                        print(f"Sample findings for {engs} {engs_cyl}:")
                        print(findings[:3])  # Show first 3 findings
                        
                        # Now follow exact main.py logic
                        fault_cat = {
                            'Blow-by in combustion chamber': {},
                            'Injection System Fault': {},
                            'Start of Injection Fault': {},
                            'Exhaust Valve Fault': {}
                        }
                        
                        for fids in f1.index:
                            for fids2 in f1.loc[fids, 'finding'].split('||'):
                                if 'Injection system fault' in fids2:
                                    fault_cat['Injection System Fault'].update({fids: fids2})
                                if 'Start of injection' in fids2:
                                    fault_cat['Start of Injection Fault'].update({fids: fids2})
                                if 'Exhaust valve' in fids2:
                                    try:
                                        fault_cat['Exhaust Valve Fault'][fids] += '||' + fids2
                                    except:
                                        fault_cat['Exhaust Valve Fault'].update({fids: fids2})
                                if 'Blow-by in combustion chamber' in fids2:
                                    fault_cat['Blow-by in combustion chamber'].update({fids: fids2})
                        
                        if fault_cats not in fault_cat or len(fault_cat[fault_cats]) == 0:
                            print(f"Skipping {fault_cats} - no faults found in this category")
                            continue
                        
                        ll = pd.DataFrame(fault_cat[fault_cats], index=['findings']).T
                        ll['Date'] = list(map(lambda x: str(x).split(' ')[0], ll.index))
                        
                        # Check severity occurrences
                        zero_chk = dict(ll['findings'].apply(lambda x: x.split('(')[1].split(')')[0] if '(' in x and ')' in x else '0').value_counts())
                        total_counts_0occ = zero_chk.get('0', 0)
                        
                        new_f = pd.DataFrame()
                        for llids in ll['Date'].unique():
                            sub_count = {}
                            sub_count['3'] = len(ll[ll['Date']==llids].loc[ll[ll['Date']==llids]['findings'].str.contains(r'\(3\)', na=False)]['findings'])
                            sub_count['2'] = len(ll[ll['Date']==llids].loc[ll[ll['Date']==llids]['findings'].str.contains(r'\(2\)', na=False)]['findings'])
                            sub_count['1'] = len(ll[ll['Date']==llids].loc[ll[ll['Date']==llids]['findings'].str.contains(r'\(1\)', na=False)]['findings'])
                            sub_count['0'] = len(ll[ll['Date']==llids].loc[ll[ll['Date']==llids]['findings'].str.contains(r'\(0\)', na=False)]['findings'])
                            new_f_sub = list(dict(sorted(sub_count.items(), key=lambda x: x[1], reverse=True)).keys())[0]
                            new_f = pd.concat([new_f, pd.DataFrame({'fault_status': new_f_sub}, index=[llids])])
                        
                        print(f"Found {len(fault_cat[fault_cats])} fault instances for {fault_cats}")
                        trigger = exists_consecutively(new_f['fault_status'], '0')

                        print(f"Severity counts: {zero_chk}")
                        print(f"Total '0' occurrences: {total_counts_0occ}")
                        print(f"Consecutive '0' trigger: {trigger}")
                        
                        # if trigger and total_counts_0occ >= 21:
                        if total_counts_0occ >= 15:
                            cyl_wise_end = {}
                            cyl_wise_end['id'] = 'maintenance_alert'
                            cyl_wise_end['uploadDate'] = upload_date
                            cyl_wise_end['measureDate'] = measured_date
                            cyl_wise_end['faultIdHat'] = utility_dict['Fault_cats_ids'][fault_cats]
                            cyl_wise_end['faultDescrHat'] = fault_cats
                            cyl_wise_end['shipCustom_1'] = 'vessel_123'
                            
                            if engs == 'Engine_1':
                                cyl_wise_end['subName'] = me1_faults[me1_faults['Fault_cat']==fault_cats]['Component1'].values[0].split('NO.')[0].strip()
                                if fault_cats != 'Injection System Fault':
                                    tt = me1_faults[(me1_faults['Fault_cat']==fault_cats)&(me1_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.'+engs_cyl.split('_')[1].strip()))]
                                else:
                                    tt = me1_faults[(me1_faults['Fault_cat']==fault_cats)&(me1_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.1'))]
                                cyl_wise_end['sysCustom_1'] = tt['EquipmentCode'].values[0]
                                cyl_wise_end['sysCustom_2'] = int(tt['EQUIPMENT_ID'].values[0])
                                cyl_wise_end['sysCustom_3'] = int(tt['Job_Plan_ID'].values[0])
                            
                            cyl_wise_end['ratingLevelHat'] = 0
                            try:
                                recc = ','.join(utility_dict['faults_recom_API']['DF'][fault_cats])
                            except KeyError:
                                recc = "No recommendations available"
                            
                            cyl_wise_end['findRecom'] = f"(0)({utility_dict['Fault_cats_ids'][fault_cats]})-{fault_cats}-{recc}"
                            api_format['surveys'].append(cyl_wise_end)
                    
                    except Exception as cyl_error:
                        print(f"Error processing cylinder {engs} {engs_cyl}: {cyl_error}")
                        continue
        
        # Remove duplicates (exact main.py logic)
        seen = []
        unique_surveys = []
        for d in api_format['surveys']:
            if 'sysCustom_1' in d and d['sysCustom_1'] not in seen:
                seen.append(d['sysCustom_1'])
                unique_surveys.append(d)
        
        api_format['surveys'] = unique_surveys
        api_format['totalRecords'] = len(api_format['surveys'])
        
        return api_format
    
    except Exception as e:
        print(f"Error in process_smart_maintenance_results: {e}")
        return {"totalRecords": 0, "surveys": [], "error": str(e)}    
    
# app = FastAPI()

# @app.post("/predict")
# async def predict():
#     try:
#         # Step 1: Get fault matrix from external API
#         fault_matrix_url = "http://192.168.18.176:4001/api/faultsense/get/?fk_vessel=1&tag1=me&tag2=fm"
#         response = requests.get(fault_matrix_url)
        
#         if response.status_code != 200:
#             return {"status": "error", "message": "Failed to fetch fault matrix"}
        
#         api_data = response.json()
        
#         if not api_data.get('success'):
#             return {"status": "error", "message": "API returned unsuccessful response"}
        
#         # Extract fault matrix data and ID
#         fault_matrix_id = api_data['data']['id']
#         fault_matrix_data = api_data['data']['data']
        
#         # Step 2: Create config with fault matrix
#         user_config = {
#             'fault_matrix': fault_matrix_data
#         }
        
#         # Step 3: Run your existing pipeline
#         fixed_data_path = r'C:\Users\User\Desktop\FaultSenseAI_source\smartmaintanace\mgd.csv'
#         result = run_pipeline(fixed_data_path, user_config)

#         # Step 4: Process results
#         today_str = datetime.now().strftime("%Y-%m-%d")
#         pickle_path = f'complete_maintenance_results_{today_str}.pickle'

#         if os.path.exists(pickle_path):
#             utility_dict_test = load_config_with_overrides(user_config)
#             maintenance_result = process_smart_maintenance_results(pickle_path, utility_dict_test)
            
#             # Step 5: Post results to external API
#             result_payload = {
#                 "status": 200,
#                 "fk_faultsenseconfig": fault_matrix_id,  # The ID we got from GET
#                 "data": maintenance_result
#             }
            
#             # Post to external API
#             post_url = "http://192.168.18.176:4001/api/faultsense/alert/"
#             post_response = requests.post(post_url, json=result_payload)
            
#             if post_response.status_code == 200:
#                 # Return success response to user
#                 return {
#                     "status": 200,
#                     "message": "Prediction completed and results posted successfully",
#                     "fault_matrix_id": fault_matrix_id,
#                     "data": maintenance_result
#                 }
#             else:
#                 # Pipeline worked but posting failed
#                 return {
#                     "status": 206,  # Partial success
#                     "message": "Prediction completed but failed to post results",
#                     "fault_matrix_id": fault_matrix_id,
#                     "data": maintenance_result,
#                     "post_error": f"Failed to post results: {post_response.status_code}"
#                 }
#         else:
#             return {"status": "error", "message": "Maintenance results not generated"}
        
#     except requests.RequestException as e:
#         return {"status": "error", "message": f"Network error: {str(e)}"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.get("/available-features")
# async def get_features():
#     return {"available": ['Pcomp_Pscav', 'PR', 'Pmax', 'Ntc', 'Ntc_Pscav', 'Pcomp', 'Pscav', 'Texh']}

# if __name__ == "__main__":
#         import uvicorn
#         uvicorn.run(app, host="0.0.0.0", port=8000)