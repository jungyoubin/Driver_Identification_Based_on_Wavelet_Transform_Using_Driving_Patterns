import pandas as pd
import csv
import pytz
from datetime import datetime, timedelta
import glob
import numpy as np
import math
import pywt
from scipy.stats import skew, kurtosis


all_df = pd.DataFrame()
name_lt = ['choimingi', 'choimingi_Auto', 'houjonguk', 'houjonguk_Auto', 'jeongyubin', 'jeongyubin_Auto', 'leegahyeon', 'leegahyeon_Auto']
# name_lt = ['houjonguk_Auto']

# name_lt = ['choimingi', 'jojeongdeok', 'leeyunguel', 'jeongyubin', 'huhongjune', 'leegahyeon', 'leegihun', 'leejaeho', 'leekanghyuk', 'simboseok', 'leeseunglee']
course = ['A', 'B', 'C']
# signal_lt = ['DI_accelPedalPos', 'DI_regenLight', 'RearMotorCurrent126', 'SteeringSpeed129', 'SteeringAngle129', 'SmoothBattCurrent132',
#              'DI_vehicleSpeed', 'RearTorqueRequest1D8', 'RearTorque1D8', 'SystemHeatPowerMax268', 'SystemHeatPower268', 'RearHeatPower266', 'RearPower266',
#              'MinVoltage2D2', 'MaxVoltage2D2', 'BMS_maxDischargePower', 'VCFRONT_chillerExvFlowm3', 'VCFRONT_pumpBatteryRPMActualm0',
#              'DIR_axleSpeed', 'DIR_torqueCommand', 'DIR_torqueActual', 'BattVoltage132']



# 윈도우 사이즈와 스트라이드 설정
window_size = 60
stride = 1

'''
########### Time-slicing ###########
'''

print('------- Time-slicing -------')

# 1초 단위로 샘플링

for name in name_lt:
    for c in course:
        path = f'../../Autopilot/{name}/{c}'
        path_lt = glob.glob(f'{path}/*DECODE*')
        for idx, i in enumerate(path_lt):
            df = pd.read_csv(i)
            output_path = f'{path}/pp_1s_{name}_{c}_{idx+1}.csv'  # 본인 경로에 맞게 수정

            res = pd.DataFrame()
            first_iteration = True

            for s in signal_lt:
                filtered_df = df[df['Signal'] == s]
                filtered_df = filtered_df.reset_index()

                filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'], format='%Y.%m.%d.%H:%M:%S.%f')
                filtered_df['timestamp_rounded'] = filtered_df['Timestamp'].dt.round('1S')
                filtered_df = filtered_df.drop_duplicates(subset='timestamp_rounded', keep='first')

                if first_iteration:
                    res['timestamp'] = filtered_df['timestamp_rounded']
                    res.set_index('timestamp', inplace=True)
                    first_iteration = False

                res = res.merge(filtered_df[['timestamp_rounded', 'Physical_value']].set_index('timestamp_rounded'),
                                how='left', left_index=True, right_index=True)
                res.rename(columns={'Physical_value': s}, inplace=True)

            res.to_csv(output_path, index=False)
            print(f'{name}_{c}_{idx + 1} --> Done')

print('-----------------------------------------------------')

# for name in name_lt:
#     path = f'../../Autopilot/CAN_extract/{name}'
#     path_lt = glob.glob(f'{path}/2023*')
#     for idx, i in enumerate(path_lt):
#         df = pd.read_csv(i)
#         output_path = f'{path}/pp_1s_{name}_{idx+1}.csv'  # 본인 경로에 맞게 수정

#         res = pd.DataFrame()
#         first_iteration = True

#         for s in signal_lt:
#             filtered_df = df[df['Signal'] == s]
#             filtered_df = filtered_df.reset_index()

#             filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'], format='%Y.%m.%d.%H:%M:%S.%f')
#             filtered_df['timestamp_rounded'] = filtered_df['Timestamp'].dt.round('1S')
#             filtered_df = filtered_df.drop_duplicates(subset='timestamp_rounded', keep='first')

#             if first_iteration:
#                 res['timestamp'] = filtered_df['timestamp_rounded']
#                 res.set_index('timestamp', inplace=True)
#                 first_iteration = False

#             res = res.merge(filtered_df[['timestamp_rounded', 'Physical_value']].set_index('timestamp_rounded'),
#                             how='left', left_index=True, right_index=True)
#             res.rename(columns={'Physical_value': s}, inplace=True)

#         res.to_csv(output_path, index=False)
#         print(f'{name}_{idx + 1} --> Done')

# print('-----------------------------------------------------')


'''
########### Normalization ###########
'''

def min_max_scaling(column):
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)
    return scaled_column

for name in name_lt:
    path = f'../../Autopilot/CAN_extract/{name}'
    path_lt = glob.glob(f'{path}/pp_*')
    for idx, i in enumerate(path_lt):
        output_path = f'{path}/nor_1s_{name}_{idx+1}.csv'  # 본인 경로에 맞게 수정

        df = pd.read_csv(i)
        columns = df.columns
        for col in columns:
            df[col] = min_max_scaling(df[col])
        # # 각 열을 Min-Max 정규화
        # df['x'] = min_max_scaling(df['x'])
        # df['y'] = min_max_scaling(df['y'])
        # df['z'] = min_max_scaling(df['z'])

        # df = df.round(4) # 소수점 4자리 이하 반올림
        # 결측치 이전값으로 대체
        df = df.fillna(method='ffill')
        df.to_csv(output_path, index=False)
        print(f'{name}_{idx + 1} --> Done')

print('Normalization --> Done')
print('-----------------------------------------------------')

'''
########### Wavelet Transform ###########
'''

# 데이터 이산 웨이블릿 변환
def wavelet_transform(data):
    # 웨이블릿 변환 수행 (레벨 1) - cA1 계산
    wavelet = 'db4'
    level = 1
    coeffs_level1 = pywt.wavedec(data, wavelet, level=level)
    cA1, cD1 = coeffs_level1[0], coeffs_level1[1]

    # 레벨 2로 웨이블릿 변환 수행 - cA2 계산
    level = 2
    coeffs_level2 = pywt.wavedec(data, wavelet, level=level)
    cA2, cD2 = coeffs_level2[0], coeffs_level2[1]
    return cA1, cA2, cD1, cD2


'''
########### Feature Extraction ###########
'''

# WE (Wavelet Energy) 계산 함수
def calculate_wavelet_energy(coefficients):
    return np.sum(np.square(coefficients))

# WEE (Wavelet Energy Entropy) 계산 함수
def calculate_we_entropy(energy_values):
    total_energy = sum(energy_values)  # 에너지 합계 계산
    if total_energy == 0:
        return 0
    # 에너지 레벨에 따른 확률분포 계산
    probabilities = [energy / total_energy for energy in energy_values]
    # Shannon's entropy 계산
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    return entropy

# Data entropy 계산 (Shannon's entropy)
def shannon_entropy(probabilities):
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += p * math.log2(1 / p)
    return entropy

# The k Percentile 계산 (k = 5, 25, 75, 95)
def percentile(data):
    percentile_5 = np.percentile(data, 5)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    percentile_95 = np.percentile(data, 95)
    return percentile_5, percentile_25, percentile_75, percentile_95


for idx, name in enumerate(name_lt):
    for idx2, c in enumerate(course):
        path = f'../../DECODE_CAN/{name}/{c}'
        path_lt = glob.glob(f'{path}/nor_1s_*')
        for idx3, i in enumerate(path_lt):
            print(len(path_lt))
            df = pd.read_csv(i)
            final_df = pd.DataFrame() # 한 회차당 dataframe
            for f in df.columns:
                features = pd.DataFrame() # 한 feature에 대한 dataframe
                DWT_result = pd.DataFrame() # WE, WEE 저장
                for w in range(0, len(df) - window_size + 1, stride):
                    window = df.iloc[w:w + window_size]
                    X = window[f]

                    # Wavelet Packet Transform 수행
                    cA1, cA2, cD1, cD2 = wavelet_transform(X.values)

                    cA1 = np.around(cA1, 4)
                    cA2 = np.around(cA2, 4)
                    cD1 = np.around(cD1, 4)
                    cD2 = np.around(cD2, 4)

                    new_data = {'cA1': [cA1], 'cA2': [cA2], 'cD1': [cD1], 'cD2': [cD2]}
                    DWT_result = pd.concat([DWT_result, pd.DataFrame(new_data)], ignore_index=True)


                # 레벨, 계수별 Wavelet 에너지 계산 및 저장
                DWT_result['cA1_energy'] = DWT_result['cA1'].apply(calculate_wavelet_energy)
                DWT_result['cA2_energy'] = DWT_result['cA2'].apply(calculate_wavelet_energy)
                DWT_result['cD1_energy'] = DWT_result['cD1'].apply(calculate_wavelet_energy)
                DWT_result['cD2_energy'] = DWT_result['cD2'].apply(calculate_wavelet_energy)
                DWT_result['WEE'] = DWT_result.apply(lambda row: calculate_we_entropy([row['cA1_energy'], row['cA2_energy'], row['cD1_energy'], row['cD2_energy']]), axis=1)

                ########################################
                features['cA1_energy'] = DWT_result['cA1_energy']
                features['cA2_energy'] = DWT_result['cA2_energy']
                features['cD1_energy'] = DWT_result['cD1_energy']
                features['cD2_energy'] = DWT_result['cD2_energy']
                features['WEE'] = DWT_result['WEE']
                ########################################



                ########## feature extraction ##########

                entropy_result = pd.DataFrame(columns=['Raw_entropy', 'cA_entropy', 'cD_entropy']) # Data entropy 저장
                for w in range(0, len(df) - window_size + 1, stride):
                    window = df.iloc[w:w + window_size]  # 60개 데이터 포인트로 이루어진 윈도우
                    X = window[f]
                    # Shannon 엔트로피 계산
                    entropy = shannon_entropy(X.values)
                    entropy_cA = shannon_entropy(np.concatenate((DWT_result['cA1'][w], DWT_result['cA2'][w])))
                    entropy_cD = shannon_entropy(np.concatenate((DWT_result['cD1'][w], DWT_result['cD2'][w])))

                    new_data = {'Raw_entropy': [entropy], 'cA_entropy': [entropy_cA], 'cD_entropy': [entropy_cD]}
                    entropy_result = pd.concat([entropy_result, pd.DataFrame(new_data)], ignore_index=True)

                ########################################
                features['Raw_entropy'] = entropy_result['Raw_entropy']
                features['cA_entropy'] = entropy_result['cA_entropy']
                features['cD_entropy'] = entropy_result['cD_entropy']
                ########################################

                Raw_percentile_result = pd.DataFrame(columns=['Raw_5_percentile', 'Raw_25_percentile', 'Raw_75_percentile', 'Raw_95_percentile'])
                for w in range(0, len(df) - window_size + 1, stride):
                    window = df.iloc[w:w + window_size]  # 60개 데이터 포인트로 이루어진 윈도우
                    X = window[f]

                    percent_5, percent_25, percent_75, percent_95 = percentile(X.values)

                    new_data = {'Raw_5_percentile': [percent_5], 'Raw_25_percentile': [percent_25],
                                'Raw_75_percentile': [percent_75], 'Raw_95_percentile': [percent_95]}
                    Raw_percentile_result = pd.concat([Raw_percentile_result, pd.DataFrame(new_data)], ignore_index=True)

                cA_percentile_result = pd.DataFrame(columns=['cA_5_percentile', 'cA_25_percentile', 'cA_75_percentile', 'cA_95_percentile']) # cA1 percentile 저장
                for n in range(len(DWT_result)):
                    X = np.concatenate((DWT_result['cA1'][n], DWT_result['cA2'][n]))

                    percent_5, percent_25, percent_75, percent_95 = percentile(X)

                    new_data = {'cA_5_percentile': [percent_5], 'cA_25_percentile': [percent_25],
                                'cA_75_percentile': [percent_75], 'cA_95_percentile': [percent_95]}
                    cA_percentile_result = pd.concat([cA_percentile_result, pd.DataFrame(new_data)], ignore_index=True)

                cD_percentile_result = pd.DataFrame(columns=['cD_5_percentile', 'cD_25_percentile', 'cD_75_percentile', 'cD_95_percentile'])
                for n in range(len(DWT_result)):
                    X = np.concatenate((DWT_result['cD1'][n], DWT_result['cD2'][n]))

                    percent_5, percent_25, percent_75, percent_95 = percentile(X)
                    
                    new_data = {'cD_5_percentile': [percent_5], 'cD_25_percentile': [percent_25],
                    'cD_75_percentile': [percent_75], 'cD_95_percentile': [percent_95]}
                    cD_percentile_result = pd.concat([cD_percentile_result, pd.DataFrame(new_data)], ignore_index=True)
                    
                    percentile_result = pd.concat([Raw_percentile_result, cA_percentile_result, cD_percentile_result], axis=1)
                    
                    
                    Raw_etc_result = pd.DataFrame(columns=['Raw_Mean', 'Raw_Median', 'Raw_Variance', 'Raw_Std_deviation', 'Raw_RMS', 'Raw_Skewness', 'Raw_Kurtosis']) # 그 외 Raw data 특징들 저장
                for w in range(0, len(df) - window_size + 1, stride):
                    window = df.iloc[w:w + window_size]
                    X = window[f]

                    # 특징 추출
                    mean = X.mean()
                    median = X.median()
                    variance = X.var()
                    std_dev = X.std()
                    rms = np.sqrt((X ** 2).mean())
                    skewness = skew(X)
                    kurt = kurtosis(X)

                    Raw_etc_result = pd.concat([Raw_etc_result, pd.DataFrame({
                    'Raw_Mean': [mean],
                    'Raw_Median': [median],
                    'Raw_Variance': [variance],
                    'Raw_Std_deviation': [std_dev],
                    'Raw_RMS': [rms],
                    'Raw_Skewness': [skewness],
                    'Raw_Kurtosis': [kurt]})], ignore_index=True)

                cA_etc_result = pd.DataFrame(columns=['cA_Mean', 'cA_Median', 'cA_Variance', 'cA_Std_deviation', 'cA_RMS', 'cA_Skewness', 'cA_Kurtosis']) # 그 외 cA 데이터 특징들 저장
                for n in range(len(DWT_result)):
                    X = np.concatenate((DWT_result['cA1'][n], DWT_result['cA2'][n]))

                    # 특징 추출
                    mean = X.mean()
                    median = np.median(X)
                    variance = X.var()
                    std_dev = X.std()
                    rms = np.sqrt((X ** 2).mean())
                    skewness = skew(X)
                    kurt = kurtosis(X)

                    cA_etc_result = pd.concat([cA_etc_result, pd.DataFrame({
                    'cA_Mean': [mean],
                    'cA_Median': [median],
                    'cA_Variance': [variance],
                    'cA_Std_deviation': [std_dev],
                    'cA_RMS': [rms],
                    'cA_Skewness': [skewness],
                    'cA_Kurtosis': [kurt]})], ignore_index=True)

                cD_etc_result = pd.DataFrame(columns=['cD_Mean', 'cD_Median', 'cD_Variance', 'cD_Std_deviation', 'cD_RMS', 'cD_Skewness', 'cD_Kurtosis']) # 그 외 cD 데이터 특징들 저장
                for n in range(len(DWT_result)):
                    X = np.concatenate((DWT_result['cD1'][n], DWT_result['cD2'][n]))

                    # 특징 추출
                    mean = X.mean()
                    median = np.median(X)
                    variance = X.var()
                    std_dev = X.std()
                    rms = np.sqrt((X ** 2).mean())
                    skewness = skew(X)
                    kurt = kurtosis(X)

                    cD_etc_result = pd.concat([cD_etc_result, pd.DataFrame({
                    'cD_Mean': [mean],
                    'cD_Median': [median],
                    'cD_Variance': [variance],
                    'cD_Std_deviation': [std_dev],
                    'cD_RMS': [rms],
                    'cD_Skewness': [skewness],
                    'cD_Kurtosis': [kurt]})], ignore_index=True)

                etc_result = pd.concat([Raw_etc_result, cA_etc_result, cD_etc_result], axis=1)

#                 ########################################
#                 features = pd.concat([features, percentile_result], axis=1)
#                 features = pd.concat([features, etc_result], axis=1)
# 
#                 features['label'] = idx
#                 features['course'] = idx2
#                 features['drive_count'] = idx3
# 
#                 print(f'{name}_{c}_{idx3+1}_{f} --> Done')
#                 ########################################
# 
#                 final_df = pd.concat([final_df, features])
# 
# final_df.to_csv('3_data_accel.csv', index=False)
                    
                ## 들여쓰기 수정 필요 ##
                ########################################
                features = pd.concat([features, percentile_result], axis=1)
                features = pd.concat([features, etc_result], axis=1)
                ########################################
                
                features.columns = [f'{f}_' + col for col in features.columns]
                final_df = pd.concat([final_df, features], axis=1)
                print(f'{name}_{c}_{idx3 + 1}_{f} --> Done')
                
            final_df['label'] = idx
            final_df['course'] = idx2
            final_df['drive_count'] = idx3
            all_df = pd.concat([all_df, final_df])
                    
                # all_df.to_csv('11_CAN_window60.csv', index=False)
all_df.to_csv('3_data_result.csv', index=False)
