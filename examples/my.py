
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%

revolution_frequency = 11245.5  # Hz
observation_points_per_turn = 550
desired_frequency = 1000.0 

# Calculate the time for one complete turn
turn_duration = 1 / revolution_frequency

num_turns = 1000

# Time
t = np.linspace(0, turn_duration*num_turns, observation_points_per_turn*num_turns, endpoint=False)

# BPMs
bpms = np.arange(1, observation_points_per_turn+1)

# BPM names
bpms_names = [f"bpm_{i}" for i in bpms.tolist()]

# Repeat for all turns
bpms_names = np.tile(bpms_names, num_turns)
print(bpms_names)

# Turns
turns = np.repeat(np.arange(num_turns), observation_points_per_turn)

# Phase
phase = np.pi/4.

signal = 10.0*np.sin(2 * np.pi * desired_frequency * t + phase)

new_phase = np.pi/2.
new_frequency = 1000.0
new_signal_noise = 0.1*np.sin(2 * np.pi * new_frequency * t + new_phase)
signal_broken = signal.copy()
bpm_broken = 550

for i in range(len(t)):
    if i%550 == 0 and not i==0:
        signal_broken[i+bpm_broken:] += new_signal_noise[i:-bpm_broken]

'''
print(len(signal_broken), len(signal))
for i in range(len(t)):
    if i%len(bpms) == 0 and i!=0:
        print(i + bpm_broken)
        
        signal_broken[i + bpm_broken:] += new_signal_noise[i:-bpm_broken]
'''
#plt.plot(signal_broken[0:550])
#plt.plot(signal[0:550])
plt.plot(signal_broken[551:551*2] - signal[551:551*2])
print(len(signal_broken), len(signal))



#%%
# Create a dictionary to store the data
data_dict = {'Turn': turns, 'BPM': bpms_names, 'Time': t,  'Signal': signal, 'Signal_broken': signal_broken}
#data_dict['Signal']

# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data_dict)

# Identify the indices where BPM 549 occurs in the DataFrame
bpm_550_indices = df[df['BPM'] == 'bpm_550'].index

# Pivot the DataFrame to have columns as the unique 550 observation points and rows as the data of each observation at each turn
df_time = df.pivot(index='Turn', columns='BPM', values='Time')

#noise = 4* np.sin(2 * np.pi * desired_frequency *  df_time[f'bpm_{bpms[-1]}'].values + phase_noise)
#df.loc[indices, 'Signal'] += noise

# Pivot the DataFrame to have columns as the unique 550 observation points and rows as the data of each observation at each turn
df_bpm_turn = df.pivot(index='Turn', columns='BPM', values='Signal')
df_bpm_turn_broken = df.pivot(index='Turn', columns='BPM', values='Signal_broken')

df_bpm_turn = df_bpm_turn.reindex(sorted(df_bpm_turn.columns, key=lambda x: int(x.split("_")[1])), axis=1)
df_bpm_turn_broken = df_bpm_turn_broken.reindex(sorted(df_bpm_turn_broken.columns, key=lambda x: int(x.split("_")[1])), axis=1)
print(df)
print(df_bpm_turn)
df_time = df_time.reindex(sorted(df_bpm_turn.columns, key=lambda x: int(x.split("_")[1])), axis=1)

print(df_time)

#signal = 10.0*np.sin(2 * np.pi * desired_frequency * t + phase)

# %%
fig,ax = plt.subplots(nrows=2, sharex=True)
fig2, ax2 = plt.subplots()
df_corrected = pd.DataFrame()

for bpm in bpms:
    #plt.plot(df_time[f"bpm_{bpm}"], df_bpm_turn[f"bpm_{bpm}"], label=f"BPM {bpm}")
    plt.sca(ax[0])
    plt.plot(df_time[f"bpm_1"], df_bpm_turn_broken[f"bpm_{bpm}"], label=f"BPM {bpm}")

    
    plt.sca(ax2)
    #current_fourier = np.fft.fft(df_bpm_turn[f"bpm_{bpm}"].values)
    current_fourier = np.fft.fft(df_bpm_turn_broken[f"bpm_{bpm}"].values)
    current_fourier = current_fourier/len(current_fourier) *2.
    #freqs = np.linspace(0, revolution_frequency, len(current_fourier)-1)
    freqs = np.fft.fftfreq(len(current_fourier), 1 / revolution_frequency)
    plt.xlim(0, 0.5*revolution_frequency)
    plt.plot(freqs, abs(current_fourier))
    Dt = df_time[f"bpm_{bpm}"].loc[0]
    corrected_fourier = current_fourier*np.exp(-1j*2.0*np.pi*freqs*Dt)
    corrected_signal = np.fft.ifft(corrected_fourier)
    corrected_signal = corrected_signal*len(corrected_signal)/2.
    plt.sca(ax[1])
    plt.plot(df_time[f"bpm_1"], corrected_signal, label=f"BPM {bpm}")
    #print(len(corrected_signal))
    df_corrected[f'corrected_signal_bpm_{bpm}'] = corrected_signal
    df_corrected[f'time{bpm}'] = df_time[f"bpm_1"]
    
    
plt.sca(ax[0])
plt.xlim(turn_duration*1,turn_duration*100)
# %%
from scipy.optimize import least_squares
import numpy as np
import scipy
fig, ax = plt.subplots()
# Define the error function for the system of equations without phase break!!
def error_function(params, t, *signals):
    A, f, phase = params
    model_signals = A * np.sin(2 * np.pi * f * t + phase)
    #print(np.mean([signal - model_signals for signal in signals], axis = 1).shape)
    return np.concatenate([signal - model_signals for signal in signals])
    #return np.mean([signal - model_signals for signal in signals], axis = 1)

# df_corrected is my DataFrame containing the signals and time data
signals =  [df_corrected[f'corrected_signal_bpm_{i}'].values for i in range(1, len(bpms) + 1)] 
print(np.array(signals).shape)

# Apply function use u=insead of the list
#df_corrected.apply(lambda x: print(x), axis = 0)
#df_corrected.drop(columns = ['time'])

fft_result = scipy.fft.fft(df_corrected['corrected_signal_bpm_1'].values)
fft_result = fft_result/len(fft_result)*2.
t = df_corrected['time1'].values  # Assuming a common time column for all signals
idx = np.argmax(abs(fft_result))
guess_amplitude = abs(fft_result)[idx]
guess_frequency = freqs[idx]
guess_phase = np.angle(fft_result[idx])

initial_guess =  [guess_amplitude, guess_frequency, guess_phase]  # In
print(initial_guess)

# Perform the optimization
result = least_squares(error_function, initial_guess, args=(t, *signals))                # Pass the signals that we 

# Extract the optimized parameters
A_opt, f_opt, phase_opt = result.x
print(result)

print(f"A = {A_opt}, f = {f_opt}, phase = {np.array(abs(np.array(phase_opt)))%np.pi}")
plt.plot(freqs, abs(fft_result))

plt.xlim(0,2000)
old_fft = abs(np.fft.fft(df_bpm_turn[f"bpm_1"].values))
old_fft = old_fft/len(old_fft)*2.
plt.plot(freqs, old_fft)# %%

# %%
# DSn in the last bpm


from scipy.optimize import least_squares
import numpy as np
import scipy
# Define the error function for the system of equations without phase break!!
#result_new = []
#result_new_all = []

def error_function_last(params, t_df, signals):
    A, f, phase, DSn = params
    for counter in range(len(signals)):
        model_signals = A * np.sin(2 * np.pi * f * t_df[counter] + phase)
        if counter == len(signals) - 1:
            result = signals[counter] - DSn - model_signals
        #elif i == len(signals) - 2:
        #    result = signals[i] -  DSn2 - model_signals
        else:
            result = signals[counter] - model_signals
    
        result_new = np.concatenate([result])
    #print(np.mean([signal - model_signals for signal in signals], axis = 1).shape)
    return result_new #np.concatenate([signal - model_signals for signal in signals])
    #return np.mean([signal - model_signals for signal in signals], axis = 1)


def error_function_all(params, t_df, signals, *mydsn):
    A, f, phase, *mydsn = params
    #A, f, phase, DSn, DSn2 = params
    #for counter in range(len(signals)):
    for counter in range(len(mydsn)):
        print(counter)
        model_signals = A * np.sin(2 * np.pi * f * t_df[counter] + phase)
        #if counter == len(signals) - 1:
        #    result = signals[counter] - DSn - DSn2 - model_signals
        #elif i == len(signals) - 2:
        #    result = signals[counter] -  DSn2 - model_signals
        #else:
            #result = signals[counter] - model_signals
        result = signals[counter] - mydsn[counter]- model_signals
    
        result_new_all = np.concatenate([result])
    #print(np.mean([signal - model_signals for signal in signals], axis = 1).shape)
    return result_new_all #np.concatenate([signal - model_signals for signal in signals])
    #return np.mean([signal - model_signals for signal in signals], axis = 1)






# df_corrected is my DataFrame containing the signals and time data
#signals = list_bpm #[df_corrected[f'corrected_signal_bpm_{i}'].values for i in range(1, len(bpms) + 1)] 

# Apply function use u=insead of the list
#df_corrected.apply(lambda x: print(x), axis = 0)
#df_corrected.drop(columns = ['time'])

fft_result = scipy.fft.fft(df_bpm_turn_broken['bpm_1'].values)
fft_result = fft_result/len(fft_result)*2.
fft_result_first = scipy.fft.fft(df_bpm_turn[f"bpm_1"].values)
fft_result_first = fft_result_first/len(fft_result_first)*2.

idx = np.argmax(abs(fft_result))
guess_amplitude = abs(fft_result)[idx]
guess_frequency = freqs[idx]
guess_phase = np.angle(fft_result[idx])

mydsn = [0] * len(bpms)
initial_guess =  [guess_amplitude, guess_frequency, guess_phase, 0.01] # In
initial_guess_all =  [guess_amplitude, guess_frequency, guess_phase] + mydsn# In
#initial_guess_all =  [guess_amplitude, guess_frequency, guess_phase, 0.01, 0.01] # In
print(initial_guess)

# Perform the optimization
result = least_squares(error_function_last, initial_guess, args=(df_time.values, df_bpm_turn_broken.values))                # Pass the signals that we 
#result_all = least_squares(error_function_all, initial_guess_all, args=(df_time.values, df_bpm_turn_broken.values))                # Pass the signals that we 

# Extract the optimized parameters
#A_opt, f_opt, phase_opt, DSn_opt, DSn2_opt = result_all.x
A_opt, f_opt, phase_opt, DSn_opt = result.x
print(result)

#print(f"A = {A_opt}, f = {f_opt}, phase = {np.array(abs(np.array(phase_opt)))%np.pi}", 'DSn_opt', DSn_opt, 'DSn2_opt', DSn2_opt)
print(f"A = {A_opt}, f = {f_opt}, phase = {np.array(abs(np.array(phase_opt)))%np.pi}", 'DSn_opt', DSn_opt)
plt.plot(freqs, abs(fft_result))
plt.plot(freqs, abs(fft_result_first))


plt.xlim(0,2000)
#old_fft = abs(np.fft.fft(df_bpm_turn[f"bpm_1"].values))
#old_fft = old_fft/len(old_fft)*2.
#plt.plot(freqs, old_fft)
# %%


# UNTIL HERE WE DID IT LAST TIME


# %%
plt.figure()
plt.plot(result_all.x[3:])
plt.axvline(bpm_broken-4)
plt.xlim(0,100)
plt.ylabel('DSn value')
plt.xlabel('BPM number')
print(result_all.x[:3], result_all.x[2]%np.pi)
plt.show()

# %%
print(result)
# %%
# Vectorized version
from scipy.optimize import least_squares
import numpy as np
import scipy
# Define the error function for the system of equations without phase break!!
#result_new = []
#result_new_all = []

def error_function_last(params, t_df, signals):
    A, f, phase, *DSn = params
    for counter in range(len(signals)):
        model_signals = A * np.sin(2 * np.pi * f * t_df[counter] + phase)
        #print(len(signals[counter]), len(DSn), len(model_signals))
        if counter == len(signals) - 1:
            result = signals[counter] - DSn - model_signals
            
        #elif i == len(signals) - 2:
        #    result = signals[i] -  DSn2 - model_signals
        else:
            result = signals[counter] - model_signals
    
        result_new = np.concatenate([result])
    #print(np.mean([signal - model_signals for signal in signals], axis = 1).shape)
    return result_new #np.concatenate([signal - model_signals for signal in signals])
    #return np.mean([signal - model_signals for signal in signals], axis = 1)


def error_function_all(params, t_df, signals, *mydsn):
    A, f, phase, *mydsn = params
    #A, f, phase, DSn, DSn2 = params
    #for counter in range(len(signals)):
    for counter in range(len(mydsn)):
        print(counter)
        model_signals = A * np.sin(2 * np.pi * f * t_df[counter] + phase)
        #if counter == len(signals) - 1:
        #    result = signals[counter] - DSn - DSn2 - model_signals
        #elif i == len(signals) - 2:
        #    result = signals[counter] -  DSn2 - model_signals
        #else:
            #result = signals[counter] - model_signals
        result = signals[counter] - mydsn[counter]- model_signals
    
        result_new_all = np.concatenate([result])
    #print(np.mean([signal - model_signals for signal in signals], axis = 1).shape)
    return result_new_all #np.concatenate([signal - model_signals for signal in signals])
    #return np.mean([signal - model_signals for signal in signals], axis = 1)

# df_corrected is my DataFrame containing the signals and time data
#signals = list_bpm #[df_corrected[f'corrected_signal_bpm_{i}'].values for i in range(1, len(bpms) + 1)] 

# Apply function use u=insead of the list
#df_corrected.apply(lambda x: print(x), axis = 0)
#df_corrected.drop(columns = ['time'])

fft_result = scipy.fft.fft(df_bpm_turn_broken['bpm_1'].values)
fft_result = fft_result/len(fft_result)*2.
fft_result_first = scipy.fft.fft(df_bpm_turn[f"bpm_1"].values)
fft_result_first = fft_result_first/len(fft_result_first)*2.

idx = np.argmax(abs(fft_result))
guess_amplitude = abs(fft_result)[idx]
guess_frequency = freqs[idx]
guess_phase = np.angle(fft_result[idx])

DSn = [0] * len(bpms)
mydsn = [0] * len(bpms)
initial_guess =  [guess_amplitude, guess_frequency, guess_phase] + DSn # In
initial_guess_all =  [guess_amplitude, guess_frequency, guess_phase] + mydsn# In
#initial_guess_all =  [guess_amplitude, guess_frequency, guess_phase, 0.01, 0.01] # In
print(initial_guess)

# Perform the optimization
result = least_squares(error_function_last, initial_guess, args=(df_time.values, df_bpm_turn_broken.values))                # Pass the signals that we 
#result_all = least_squares(error_function_all, initial_guess_all, args=(df_time.values, df_bpm_turn_broken.values))                # Pass the signals that we 

# Extract the optimized parameters
#A_opt, f_opt, phase_opt, DSn_opt, DSn2_opt = result_all.x
A_opt, f_opt, phase_opt, DSn_opt = result.x
print(result)

#print(f"A = {A_opt}, f = {f_opt}, phase = {np.array(abs(np.array(phase_opt)))%np.pi}", 'DSn_opt', DSn_opt, 'DSn2_opt', DSn2_opt)
print(f"A = {A_opt}, f = {f_opt}, phase = {np.array(abs(np.array(phase_opt)))%np.pi}", 'DSn_opt', DSn_opt)
plt.plot(freqs, abs(fft_result))
plt.plot(freqs, abs(fft_result_first))
plt.xlim(0,2000)
# %%
