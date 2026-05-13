import re
from matplotlib import cm
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import exp, linregress


# ------------------------------------------------------------
# 0. Read and process data functions
# ------------------------------------------------------------
def parse_position_filename(filename):
    """Extract X and Y coordinates from filename like Position_X_2.00_Y_5.00.txt."""
    match = re.search(r'Position_X_([\d\.]+)_Y_([\d\.]+)\.txt', filename.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename.name}")
    x = float(match.group(1))
    y = float(match.group(2))
    return x, y

def load_measurement_data(file_path):
    """
    Load a measurement file.
    Assumes two columns (frequency, value) separated by whitespace.
    Returns (freqs, values) arrays.
    If the file contains only one column, set freqs=None.
    """
    data = np.loadtxt(file_path, delimiter=None)
    if data.ndim == 1:
        # Only one column – assume it's the measured values
        return data
    else:
        raise ValueError(f"Unexpected number of columns in {file_path}")

def read_experiments(root_dir, manual_freqs=None):
    """
    Traverse root_dir for folders matching '*_Experimental_data'.
    For each folder, load all Position_X_*.txt files.
    Returns a dictionary with experiment timestamps as keys.
    Each value is a dict with:
        'positions': np.array of shape (N, 2) (x, y)
        'frequencies': np.array (if consistent across files, else None)
        'values': np.array of shape (N, n_freqs) (values per position)
    """
    experiments = {}
    root = Path(root_dir)


    # Find all measurement files
    measurement_files = [f for f in root.iterdir() if f.is_file() and "Position_X_" in f.name and f.suffix == '.txt']
    if not measurement_files:
        print(f"Warning: No measurement files found in {root}")
        return experiments
    # Store data for this experiment
    positions = []
    values = []
    # Data loading loop
    for mfile in sorted(measurement_files, key=lambda f: parse_position_filename(f)):
        try:
            x, y = parse_position_filename(mfile)
            v = load_measurement_data(mfile)
            freqs = manual_freqs         #Freq from manual input
            positions.append((x, y))    #Write position to array
            values.append(v)            #Write values to array
        except Exception as e:
            print(f"Error reading {mfile}: {e}")

    # Convert to arrays
    positions = np.array(positions)          # shape (N, 2)
    values = np.array(values)                # shape (N, n_freqs)

    # Store in experiments dictionary
    experiments = {
        'positions': positions,
        'frequencies': freqs,
        'values': values,
    }
    return experiments


# ------------------------------------------------------------
# 1. Inverted Lorentzian model for 1D (frequency vs. value)
# ------------------------------------------------------------
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
    gamma_guess = (freqs[-1] - freqs[0]) / 5
    # A: depth (baseline - min)
    baseline_guess = np.median(values[0:5])  # assume first few points are baseline
    A_guess = baseline_guess - np.min(values)
    B_guess = baseline_guess
    # Initial parameter array
    p0 = [f0_guess, gamma_guess, A_guess, B_guess]

    try:
        popt, pcov = curve_fit(inverted_lorentzian_1d, freqs, values, p0=p0)
        f0_fit = popt[0]
        f0_err = np.sqrt(pcov[0, 0])
        return f0_fit, f0_err
    except (RuntimeError, ValueError):
        return np.nan, np.nan
    
# ------------------------------------------------------------
# 2. Import experiment data
# ------------------------------------------------------------

start_freq = 2.4e9   # 2.5 GHz
stop_freq  = 3.0e9   # 2.9 GHz
n_points = 151
# Generate frequency array manually since not reading from files
manual_freqs = np.linspace(start_freq, stop_freq, n_points)

# Path to the directory containing the experiment folders
data_root = '"C:\\School Code\\Magnet Scan S\\20260205_1537_Experiment_data"'   # <-- CHANGE THIS

# Read all experiments
data = read_experiments(data_root, manual_freqs=manual_freqs)



# ============================================================
# 3. Fit dips and analyze results
# ============================================================
#Prepair Empty Arrays to store results for all experiments
dip_freqs_list = []
dip_errs_list = []
X_position_list = []
Y_position_list = []

positions = data['positions']          # shape (N, 2)
values = data['values']                # shape (N, n_freqs)
freqs = data['frequencies']            # 1D array of frequencies

N_positions = positions.shape[0]
dip_freqs = np.zeros(N_positions)
dip_errs = np.zeros(N_positions)

for i in range(N_positions):
    f0, err = fit_dip_per_position(freqs, values[i, :])
    dip_freqs[i] = f0
    dip_errs[i] = err
    dip_freqs_list = np.append(dip_freqs_list, f0)
    dip_errs_list = np.append(dip_errs_list, err)
    X_position_list = np.append(X_position_list, positions[i, 0])
    Y_position_list = np.append(Y_position_list, positions[i, 1])
    print(positions[i])


dip_freqs_2D = dip_freqs.reshape(9,9)
dip_freqs_2D = dip_freqs_2D/1e9
dip_freqs_2D = np.rot90(dip_freqs_2D, k=2, axes=(1, 0))
dip_errs_2D = dip_errs.reshape(9,9)
dip_errs_2D = dip_errs_2D/1e9
dip_errs_2D = np.rot90(dip_errs_2D, k=2, axes=(1, 0))

X_position_2D = X_position_list.reshape(9,9)
Y_position_2D = Y_position_list.reshape(9,9)

# Define threshold and replacement strategy
# threshold = 2.420   # Hz – adjust to your needs
# replacement = stop_freq/1e9  # Hz – adjust to your needs
# dip_freqs_2D[dip_freqs_2D < threshold] = replacement
# up_threshold = 3.1
# dip_freqs_2D[dip_freqs_2D > up_threshold] = replacement

# slope = 0.00026920566836065746
# Intercept = 2.581428641207388
# dip_field_2D = (dip_freqs_2D-Intercept) / slope
# dip_errs_field_2D = dip_errs_2D / slope

fig = plt.figure(figsize=(2, 5))
# ax = fig.add_subplot(projection='3d')
plt.imshow(dip_freqs_2D, extent=(Y_position_2D.min(), 75, X_position_2D.min(), 25), origin='lower',cmap='grey', aspect='auto')
plt.colorbar(label='Magnetic Field (Gs)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('CITYU sample(Partial)-16x resolution')
# ax.plot_surface(X_position_2D, Y_position_2D, dip_freqs_2D, cmap=cm.coolwarm,
#                        linewidth=1, antialiased=False)
plt.tight_layout()
plt.show()
