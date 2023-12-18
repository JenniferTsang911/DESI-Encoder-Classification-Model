import numpy as np
import pandas as pd

def tic_normalize(peaks):
    tot_ion_cur = np.sum(peaks, axis=1)
    peaks_ticn = np.empty(peaks.shape)
    for i in range(len(peaks)):
        if tot_ion_cur[i]!=0:
            peaks_ticn[i] = peaks[i]/tot_ion_cur[i]
    return peaks_ticn

# normalize each ion in whole data to have min of 0 and max of 1 (ion based normalization)
def ion_minmax_normalize(peaks):
    max_ion_int = np.max(peaks, axis=0)
    min_ion_int = np.min(peaks, axis=0)
    peaks_ionorm = np.empty(peaks.shape)
    for i in range(peaks.shape[1]):
        if max_ion_int[i]!=min_ion_int[i]:
            peaks_ionorm[:,i] = (peaks[:,i]-min_ion_int[i])/(max_ion_int[i]-min_ion_int[i])
    return peaks_ionorm

df = pd.read_csv(r"C:\Users\jenni\OneDrive - Queen's University\DESI project\DESI TXT colon\Annotated Dataset\2021 03 31 colon 1258561-2 all_aligned.csv", index_col=0)

peak_start_col = 4
mz = np.array(df.columns[peak_start_col:], dtype='float')
peaks = df[df.columns[peak_start_col:]].values

labels =  df[df.columns[:peak_start_col]].values 

peaks = np.nan_to_num(peaks)

peaks = tic_normalize(peaks)
peaks_norm = ion_minmax_normalize(peaks)

csv_data = np.concatenate( [labels, peaks] , axis=1)
csv_column = list(df.columns[:peak_start_col])+list(mz)
df = pd.DataFrame(csv_data, columns=csv_column)

print(df.head())

df.to_csv('all_aligned_processed', index=False)