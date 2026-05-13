import re
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = np.loadtxt(r"C:\School Code\20260209Experiment\20260209_1607_Experiment_data\Position_X_4.00_Y_8.00.txt")


def inverted_lorentzian_1d(f, f0, gamma, A, B):
    """Inverted Lorentzian dip: B - A / (1 + ((f - f0)/gamma)**2)"""
    return B - A / (1 + ((f - f0)/gamma)**2)

def fit_dip_per_position(freqs, values):
    """
    Fit an inverted Lorentzian to a single position's data.
    Returns (f0, f0_err) – the dip frequency and its standard error.
    """
    # Provide initial guesses (rough)
    # f0: frequency of the minimum value
    idx_min = np.argmin(values)
    f0_guess = freqs[idx_min]
    # gamma: half width at half max – estimate from FWHM
    # rough: difference between frequencies where value is halfway from min to baseline
    # simpler: take a fraction of the range
    gamma_guess = (freqs[-1] - freqs[0]) / 10
    # A: depth (baseline - min)
    baseline_guess = np.median(values[0:20])  # assume first few points are baseline
    A_guess = baseline_guess - np.min(values)
    B_guess = baseline_guess
    # Initial parameter array
    p0 = [f0_guess, gamma_guess, A_guess, B_guess]

    try:
        popt, pcov = curve_fit(inverted_lorentzian_1d, freqs, values, p0=p0)
        f0_fit = popt[0]
        f0_err = np.sqrt(pcov[0, 0])
        return f0_fit, f0_err, popt
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.array([np.nan, np.nan, np.nan, np.nan])
    
freq = np.linspace(2000000000, 2600000000, 151)

# Example usage for a single position (replace with loop for all positions
f0, f0_err, popt = fit_dip_per_position(freq, data)
print(f"Fitted dip frequency: {f0:.2f} Hz ± {f0_err:.2f} Hz")

#plot the data and the fit
plt.figure(figsize=(10, 6))
plt.plot(freq/1e9, data, 'b.', label=f'Experimental Data\nReaonance at {f0/1e9:.3f} GHz ± {f0_err/1e9:.5f} GHz\nMinimum point at {freq[np.argmin(data)]/1e9:.3f} GHz')
f_fit = np.linspace(freq[0], freq[-1], 1000)
plt.plot(f_fit/1e9, inverted_lorentzian_1d(f_fit, *popt), 'r-', label='Fit')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Transmission')
plt.title('Inverted Lorentzian Fit of VNA Scan')
plt.tight_layout()
plt.legend()
plt.show()


