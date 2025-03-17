import numpy as np

def generate_sinusoid(fs, f1, f2, A1, A2, duration):
    """
    Generate a sinusoid with two frequency components.

    Parameters:
    fs (float): Sampling frequency in Hz
    f1 (float): Frequency of the first component in Hz
    f2 (float): Frequency of the second component in Hz
    A1 (float): Amplitude of the first component
    A2 (float): Amplitude of the second component
    duration (float): Duration of the signal in seconds

    Returns:
    tuple: (t, y) where t is the time array and y is the generated signal
    """

    # Generate time array
    t = np.arange(0, duration, 1/fs)

    # Generate the two sinusoidal components
    y1 = A1 * np.sin(2 * np.pi * f1 * t)
    y2 = A2 * np.sin(2 * np.pi * f2 * t)

    # Combine the components
    y = y1 + y2

    return t, y

def generate_syth_signal(sampling_frequency, exponent,duration,dur,bstart):
    """
    Generate a synthetic signal with a smoothly transitioning burst.

    Parameters:
    sampling_frequency (int): Sampling frequency in Hz
    exponent (float): Exponent for the amplitude transition
    duration (float): Duration of the signal in seconds

    Returns:
    tuple: (time, signal) where time is the time array and signal is the generated signal
    """

    # Generate time array
    time = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)

    # Initialize signal with baseline
    signal = np.ones_like(time) * 900  # Baseline (mV)

    # Define the burst start and end times (random length burst)
    burst_start = bstart  # Start at 50 ms
    burst_duration = np.random.uniform(dur[0],dur[1])  # Random duration between 200 ms and 300 ms
    burst_end = burst_start + burst_duration

    # Create a mask for the burst region
    burst_mask = (time >= burst_start) & (time <= burst_end)
    burst_time = time[burst_mask]

    # Define the transition parameters
    freq_min = 5  # Low frequency (Hz)
    freq_max = 50  # High frequency (Hz)
    amp_min = 15  # Minimum amplitude (mV)
    amp_max = 50  # Maximum amplitude (mV)

    # Generate the frequency and amplitude transition
    freq = np.linspace(freq_min, freq_max, len(burst_time))
    amp = amp_max * (freq / freq_min) ** (-exponent)

    # Generate the smoothly transitioning burst
    burst_signal = amp * np.sin(2 * np.pi * freq * burst_time)

    # Add the burst signal to the overall signal
    signal[burst_mask] += burst_signal

    # Add random noise to the entire signal
    noise = np.random.normal(0, 2, len(time))
    signal += noise

    return time, signal
