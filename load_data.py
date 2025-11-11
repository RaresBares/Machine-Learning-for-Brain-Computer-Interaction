# Author: Thorir Mar Ingolfsson
# Use either convert_data_biowolf_GUI_2c or convert_data_biowolf_GUI_2c_new
# Both work similarly but the new version is more efficient in time.

import numpy as np
import struct
import os
import pandas as pd
def convert_data_biowolf_GUI_2c_new(file_including_path, VoltageScale, TimeStampScale):
    # Constants
    lsb_vals = {
        1: 7000700, 
        2: 14000800, 
        3: 20991300, 
        4: 27990100, 
        6: 41994600, 
        8: 55994200, 
        12: 83970500
    }
    HEADER_size = 7
    bt_pck_size = 32

    # Open, read, and close file
    with open(file_including_path, 'rb') as fileID:
        A = np.frombuffer(fileID.read(), dtype=np.uint8)
        
    # Check time and voltage scales
    tscaleFactor = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(TimeStampScale, None)
    if tscaleFactor is None:
        raise ValueError('Invalid time scale')
    
    vscaleFactor = {'V': 1, 'mV': 1e3, 'uV': 1e6, 'none': 1}.get(VoltageScale, None)
    if vscaleFactor is None:
        raise ValueError('Invalid voltage scale')
        
    # Initialize data-raw structure
    ExGData = {}
    # Read experimental notes
    # Convert the array to bytes
    bytes_data = A.tobytes()

    # Define the target sequence as bytes
    target_sequence = bytes([60, 60, 62, 62, 73, 69, 80, 44])

    # Find the index of the sequence
    index = bytes_data.find(target_sequence)

    if index != -1:
        # Recover the data-raw starting from the index
        data_recovered = bytes_data[index:].decode('utf-8')
        end_of_data = index - 1
    else:
        data_recovered = ''
        end_of_data = 0

    # Populate ExGData if headers are found
    if end_of_data != 0:
        Rparams = data_recovered.split(',')
        for t_value in Rparams[1:]:
            key = t_value[0]
            val = t_value[1:]
            if key == 'T':
                ExGData['TestName'] = val
            elif key == 'S':
                ExGData['SubjectName'] = val
            elif key == 'A':
                ExGData['SubjectAge'] = float(val) if val else np.nan
            elif key == 'R':
                ExGData['Remarks'] = val
            elif key == 'F':
                ExGData['SampleRate'] = float(val) if val else np.nan
            elif key == 'G':
                ExGData['SignalGain'] = float(val) if val else np.nan
    else:
        ExGData['SampleRate'] = 500
        ExGData['SignalGain'] = 12
        vscaleFactor = 1
        end_of_data=len(A)


    
    # Assuming A, end_of_data, and bt_pck_size are predefined
    if len(A[:end_of_data]) % bt_pck_size != 0:
        pad_length = bt_pck_size - (len(A[:end_of_data]) % bt_pck_size)
        padded_array = np.pad(A[:end_of_data], (0, pad_length), 'constant')
    else:
        padded_array = A[:end_of_data]

    # Reshape the array
    ADS = np.reshape(padded_array, (-1, bt_pck_size))

    ADS = ADS.astype(np.int32)

    ch11 = ADS[:, 0] * 256**3 + ADS[:, 1] * 256**2 + ADS[:, 2] * 256
    ch22 = ADS[:, 3] * 256**3 + ADS[:, 4] * 256**2 + ADS[:, 5] * 256
    ch33 = ADS[:, 6] * 256**3 + ADS[:, 7] * 256**2 + ADS[:, 8] * 256
    ch44 = ADS[:, 9] * 256**3 + ADS[:, 10] * 256**2 + ADS[:, 11] * 256
    ch55 = ADS[:, 12] * 256**3 + ADS[:, 13] * 256**2 + ADS[:, 14] * 256
    ch66 = ADS[:, 15] * 256**3 + ADS[:, 16] * 256**2 + ADS[:, 17] * 256
    ch77 = ADS[:, 18] * 256**3 + ADS[:, 19] * 256**2 + ADS[:, 20] * 256
    ch88 = ADS[:, 21] * 256**3 + ADS[:, 22] * 256**2 + ADS[:, 23] * 256

    # Initialize the acc array
    acc = np.zeros((len(ADS), 3), dtype=np.int32)

    # Calculate the acc values using vectorized operations
    acc[:, 0] = np.int32(ADS[:, 25] * 256**3 + ADS[:, 26] * 256**2)
    acc[:, 1] = np.int32(ADS[:, 27] * 256**3 + ADS[:, 28] * 256**2)
    acc[:, 2] = np.int32(ADS[:, 29] * 256**3 + ADS[:, 30] * 256**2)
    # Vectorized conversion from bytes to integers for channels and acceleration
    # Assuming the packet format is consistent across all packets

    # Convert ADC data-raw into volts
    if VoltageScale != 'none':
        switch = {
            0: 1,
            1: 1 / lsb_vals[1],
            2: 1 / lsb_vals[2],
            3: 1 / lsb_vals[3],
            4: 1 / lsb_vals[4],
            6: 1 / lsb_vals[6],
            8: 1 / lsb_vals[8],
            12: 1 / lsb_vals[12],
        }
        gain_scaling = switch.get(ExGData['SignalGain'], 1)
    else:
        gain_scaling = 1

    # Assuming `channels` is a 2D array where each column represents a channel (ch11, ch22, ..., ch88)
    channels = np.array([ch11, ch22, ch33, ch44, ch55, ch66, ch77, ch88]).T

    # Convert all channels at once
    t_data = (channels.astype(np.float64) / 256) * gain_scaling * vscaleFactor

    ADS = np.vstack(ADS)  # Stacks arrays vertically to create a 2D array
    t_trigger = ADS[:, 31]

    # Skip the first sample
    skipped_samples = 1
    t_trigger[-1] = 10
    # Populate ExGData (which can be a dictionary or custom object)
    #ExGData = {}
    ExGData['Data'] = t_data[skipped_samples:, :]
    ExGData['Trigger'] = t_trigger[skipped_samples:]
    ExGData['timestamp'] = np.arange(0,
                                    (t_data.shape[0] - skipped_samples) / (ExGData['SampleRate'] / tscaleFactor),
                                    tscaleFactor / ExGData['SampleRate'])
    ExGData['ImuData'] = acc / (255 * 255)

    return ExGData
def convert_data_biowolf_GUI_2c(file_including_path, VoltageScale, TimeStampScale):
    # Constants
    lsb_vals = {
        1: 7000700,
        2: 14000800,
        3: 20991300,
        4: 27990100,
        6: 41994600,
        8: 55994200,
        12: 83970500
    }
    HEADER_size = 7
    bt_pck_size = 32

    # Open, read, and close file
    with open(file_including_path, 'rb') as fileID:
        A = np.frombuffer(fileID.read(), dtype=np.uint8)

    # Check time and voltage scales
    tscaleFactor = {'s': 1, 'ms': 1e3, 'us': 1e6}.get(TimeStampScale, None)
    if tscaleFactor is None:
        raise ValueError('Invalid time scale')

    vscaleFactor = {'V': 1, 'mV': 1e3, 'uV': 1e6, 'none': 1}.get(VoltageScale, None)
    if vscaleFactor is None:
        raise ValueError('Invalid voltage scale')

    # Initialize data-raw structure
    ExGData = {}
    # Read experimental notes
    # Convert the array to bytes
    bytes_data = A.tobytes()

    # Define the target sequence as bytes
    target_sequence = bytes([60, 60, 62, 62, 73, 69, 80, 44])

    # Find the index of the sequence
    index = bytes_data.find(target_sequence)

    if index != -1:
        # Recover the data-raw starting from the index
        data_recovered = bytes_data[index:].decode('utf-8')
        end_of_data = index - 1
    else:
        data_recovered = ''
        end_of_data = 0

    # Populate ExGData if headers are found
    if end_of_data != 0:
        Rparams = data_recovered.split(',')
        for t_value in Rparams[1:]:
            key = t_value[0]
            val = t_value[1:]
            if key == 'T':
                ExGData['TestName'] = val
            elif key == 'S':
                ExGData['SubjectName'] = val
            elif key == 'A':
                ExGData['SubjectAge'] = float(val) if val else np.nan
            elif key == 'R':
                ExGData['Remarks'] = val
            elif key == 'F':
                ExGData['SampleRate'] = float(val) if val else np.nan
            elif key == 'G':
                ExGData['SignalGain'] = float(val) if val else np.nan
    else:
        ExGData['SampleRate'] = 500
        ExGData['SignalGain'] = 12
        vscaleFactor = 1e6
        end_of_data=len(A)




    # Assuming A, end_of_data, and bt_pck_size are predefined
    if len(A[:end_of_data]) % bt_pck_size != 0:
        pad_length = bt_pck_size - (len(A[:end_of_data]) % bt_pck_size)
        padded_array = np.pad(A[:end_of_data], (0, pad_length), 'constant')
    else:
        padded_array = A[:end_of_data]

    # Reshape the array
    ADS = np.reshape(padded_array, (-1, bt_pck_size))

    ADS = ADS.astype(np.int32)

    ch11 = ADS[:, 0] * 256**3 + ADS[:, 1] * 256**2 + ADS[:, 2] * 256
    ch22 = ADS[:, 3] * 256**3 + ADS[:, 4] * 256**2 + ADS[:, 5] * 256
    ch33 = ADS[:, 6] * 256**3 + ADS[:, 7] * 256**2 + ADS[:, 8] * 256
    ch44 = ADS[:, 9] * 256**3 + ADS[:, 10] * 256**2 + ADS[:, 11] * 256
    ch55 = ADS[:, 12] * 256**3 + ADS[:, 13] * 256**2 + ADS[:, 14] * 256
    ch66 = ADS[:, 15] * 256**3 + ADS[:, 16] * 256**2 + ADS[:, 17] * 256
    ch77 = ADS[:, 18] * 256**3 + ADS[:, 19] * 256**2 + ADS[:, 20] * 256
    ch88 = ADS[:, 21] * 256**3 + ADS[:, 22] * 256**2 + ADS[:, 23] * 256

    # Initialize the acc array
    acc = np.zeros((len(ADS), 3), dtype=np.int32)

    # Calculate the acc values using vectorized operations
    acc[:, 0] = np.int32(ADS[:, 25] * 256**3 + ADS[:, 26] * 256**2)
    acc[:, 1] = np.int32(ADS[:, 27] * 256**3 + ADS[:, 28] * 256**2)
    acc[:, 2] = np.int32(ADS[:, 29] * 256**3 + ADS[:, 30] * 256**2)
    # Vectorized conversion from bytes to integers for channels and acceleration
    # Assuming the packet format is consistent across all packets

    # Convert ADC data-raw into volts
    if VoltageScale != 'none':
        switch = {
            0: 1,
            1: 1 / lsb_vals[1],
            2: 1 / lsb_vals[2],
            3: 1 / lsb_vals[3],
            4: 1 / lsb_vals[4],
            6: 1 / lsb_vals[6],
            8: 1 / lsb_vals[8],
            12: 1 / lsb_vals[12],
        }
        gain_scaling = switch.get(ExGData['SignalGain'], 1)
    else:
        gain_scaling = 1

    # Assuming `channels` is a 2D array where each column represents a channel (ch11, ch22, ..., ch88)
    channels = np.array([ch11, ch22, ch33, ch44, ch55, ch66, ch77, ch88]).T

    # Convert all channels at once
    t_data = (channels.astype(np.float64) / 256) * gain_scaling * vscaleFactor

    print("File: ", file_including_path)
    print("Sample Rate: ", ExGData['SampleRate'])
    print("Signal Gain: ", ExGData['SignalGain'])
    print("Voltage Scale Factor: ", vscaleFactor)
    print("Time Scale Factor: ", tscaleFactor)
    print("Gain Scaling: ", gain_scaling)

    ADS = np.vstack(ADS)  # Stacks arrays vertically to create a 2D array
    t_trigger = ADS[:, 31]

    # Skip the first sample
    skipped_samples = 1
    t_trigger[-1] = 10
    # Populate ExGData (which can be a dictionary or custom object)
    #ExGData = {}
    ExGData['Data'] = t_data[skipped_samples:, :]
    ExGData['Trigger'] = t_trigger[skipped_samples:]
    ExGData['timestamp'] = np.arange(0,
                                    (t_data.shape[0] - skipped_samples) / (ExGData['SampleRate'] / tscaleFactor),
                                    tscaleFactor / ExGData['SampleRate'])
    ExGData['ImuData'] = acc / (255 * 255)

    return ExGData


def convert_raw_to_csv(folder_path, time_scale, voltage_scale, save_path):
    # Make save_path if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    listing = os.listdir(folder_path)
    listing.sort()

    for file in listing:

        #if(os.path.exists(save_path + file[:-3] + 'csv')):
        #    continue
        file_path = folder_path + file
        ExG_data = convert_data_biowolf_GUI_2c(file_path, voltage_scale, time_scale)

        # Assuming ExG_data is a dictionary with keys 'Data' and 'Trigger'
        T = pd.DataFrame({
            "EEG 1": ExG_data['Data'][:, 0],
            "EEG 2": ExG_data['Data'][:, 1],
            "EEG 3": ExG_data['Data'][:, 2],
            "EEG 4": ExG_data['Data'][:, 3],
            "EEG 5": ExG_data['Data'][:, 4],
            "EEG 6": ExG_data['Data'][:, 5],
            "EEG 7": ExG_data['Data'][:, 6],
            "EEG 8": ExG_data['Data'][:, 7],
            "Trigger": ExG_data['Trigger']
        })
        # Convert DataFrame columns to integers
        for col in T.columns:
            T[col] = T[col].astype(float)

        T.to_csv(save_path + file[:-3] + 'csv', index=False)
