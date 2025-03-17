import numpy as np

def delta_modulation_synced(inputSignal, ON_Threshold, OFF_Threshold):
    pulseTrain = np.zeros((2, 0))
    lastResetVoltage = 0
    for i in range(len(inputSignal)):
        diff = inputSignal[i] - lastResetVoltage
        if diff > ON_Threshold:
            num_pulse = np.floor_divide(diff,ON_Threshold).astype(int)
            pulseTrain = np.hstack((pulseTrain, np.vstack((i,num_pulse))))
            lastResetVoltage = lastResetVoltage + ON_Threshold * num_pulse
        elif diff < OFF_Threshold:
            num_pulse = np.floor_divide(diff,OFF_Threshold).astype(int)
            pulseTrain = np.hstack((pulseTrain, np.vstack((i,num_pulse * -1))))
            lastResetVoltage = lastResetVoltage + OFF_Threshold * num_pulse
    return pulseTrain

def LCADC_reconstruct(pulseTrain,Threshold):
    reconstructed_signal =  np.zeros([2,np.size(pulseTrain[0])])
    reconstructed_signal[0] = pulseTrain[0]
    reconstructed_signal[1][0] = 0
    for i in range(1,np.size(pulseTrain[0])):
            reconstructed_signal[1][i] = reconstructed_signal[1][i-1] + pulseTrain[1][i-1] * Threshold
    sig_length = np.max(pulseTrain[0]).astype(int)

    for j in range(sig_length+1):
        if j not in pulseTrain[0]:
            idx = np.where((reconstructed_signal[0] > j))
            idx = idx[0][0]
            reconstructed_signal = np.insert(reconstructed_signal,idx,[j,reconstructed_signal[1][idx]],axis=1)
    return reconstructed_signal

def evaluate_SPD(GT_spike_times, detected_spikeTimes, toleranceWindow, samplingInterval):
    trueDetection = 0
    falseDetection = 0
    missedDetection = 0
    halfToleranceWinSamples = int(toleranceWindow * 1e-3 / samplingInterval)  # tolerance +/- t ms
    spike_times_toleranceIncluded = np.concatenate([np.arange(gt - halfToleranceWinSamples, gt + halfToleranceWinSamples + 1) for gt in GT_spike_times])
    spike_times_toleranceIncluded = np.unique(spike_times_toleranceIncluded)
    
    trueDetection = np.sum(np.isin(detected_spikeTimes, spike_times_toleranceIncluded))
    falseDetection = len(detected_spikeTimes) - trueDetection
    
    spikeTimes_toleranceIncluded = np.concatenate([np.arange(dt - halfToleranceWinSamples, dt + halfToleranceWinSamples + 1) for dt in detected_spikeTimes])
    spikeTimes_toleranceIncluded = np.unique(spikeTimes_toleranceIncluded)
    
    missedDetection = np.sum(np.isin(GT_spike_times, spikeTimes_toleranceIncluded, invert=True))
    
    TP = trueDetection
    FP = falseDetection
    FN = missedDetection
    
    if TP == 0:
        return 0, 1, 0
    else:
        TPR = TP / (TP + FN)
        Sensitivity = TPR
        FDR = FP / (TP + FP)
        Accuracy = TP / (TP + FP + FN)
        return Sensitivity, FDR, Accuracy