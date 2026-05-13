import re
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress

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

def read_all_experiments(root_dir, manual_freqs=None):
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

    for folder in root.iterdir():
        timestamp = folder.name.split('_Experimental_data')[0]

        # Find all measurement files
        measurement_files = [f for f in folder.iterdir() if f.is_file() and "Position_X_" in f.name and f.suffix == '.txt']
        if not measurement_files:
            print(f"Warning: No measurement files found in {folder}")
            continue
        # Find Experiment parameters file (if exists)
        parameters_file = [f for f in folder.iterdir() if f.is_file() and "experiment_params.txt" in f.name]
        if not parameters_file:
            print(f"Warning: No experiment_params.txt found in {folder}. Skipping parameter extraction.")

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

        # Attempt to read experiment parameters (e.g., B_surface) from experiment_params.txt
        if parameters_file:    
            with open(parameters_file[0]) as f:
                for line in f:
                    if "B_surface=" in line:
                        # Split at '=' then take the part before 'Gs'
                        value_str = line.split("B_surface=")[1].split("G")[0].strip()
                        field = int(value_str)  # convert to integer
        # Store in experiments dictionary
        experiments[timestamp] = {
            'positions': positions,
            'frequencies': freqs,
            'values': values,
            'B_surface': field
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
        return f0_fit, f0_err
    except (RuntimeError, ValueError):
        return np.nan, np.nan

# ------------------------------------------------------------
# 2. 2D Gaussian model
# ------------------------------------------------------------
def inverted_gaussian_2d(xy, x0, y0, sigma_x, sigma_y, A, B):
    """
    Inverted 2D Gaussian (dip).
    xy is a tuple (x, y) of flattened arrays.
    """
    x, y = xy
    return B - A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) +
                            (y - y0)**2 / (2 * sigma_y**2)))


def gaussian_2d(xy, x0, y0, sigma_x, sigma_y, A, B):
    """Upright 2D Gaussian peak."""
    x, y = xy
    return B + A * np.exp(-((x - x0)**2 / (2 * sigma_x**2) +
                            (y - y0)**2 / (2 * sigma_y**2)))



def fit_global_minimum_freq(x_grid, y_grid, dip_freqs, dip_errs):
    """
    Fit an inverted 2D Gaussian to the grid of dip frequencies.
    Returns (f_min, f_min_err) -> the minimum frequency value and its standard error.
    Also returns (x0, y0) for reference.
    """
    # Flatten
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = dip_freqs.flatten()
    sigma_flat = dip_errs.flatten()

    # Initial guesses (based on data)
    idx_min = np.argmin(z_flat)
    x0_guess = x_flat[idx_min]
    y0_guess = y_flat[idx_min]
    # Width estimates: half the range of positions
    sigma_x_guess = (np.max(x_flat) - np.min(x_flat)) / 2
    sigma_y_guess = (np.max(y_flat) - np.min(y_flat)) / 2
    # Amplitude and baseline
    baseline_guess = np.max(z_flat)
    A_guess = baseline_guess - np.min(z_flat)
    B_guess = baseline_guess

    p0 = [x0_guess, y0_guess, sigma_x_guess, sigma_y_guess, A_guess, B_guess]

    try:
        popt, pcov = curve_fit(inverted_gaussian_2d,
                               (x_flat, y_flat), z_flat,
                               p0=p0, sigma=sigma_flat, absolute_sigma=True)

        x0, y0, sigma_x, sigma_y, A, B = popt
        f_min = B - A
        # Error propagation for f_min
        var_A = pcov[4, 4]
        var_B = pcov[5, 5]
        cov_AB = pcov[4, 5]
        f_min_err = np.sqrt(var_B + var_A - 2 * cov_AB)

        return f_min, f_min_err, (x0, y0)
    except (RuntimeError, ValueError) as e:
        print(f"Gaussian fit failed: {e}")
        return np.nan, np.nan, (np.nan, np.nan)

def fit_global_maximum_freq(x_grid, y_grid, dip_freqs, dip_errs):
    """
    Fit a 2D inverted Gaussian to the grid of dip frequencies.
    Returns (f_max, f_max_err) -> the maximum frequency value and its standard error.
    Also returns (x0, y0) for information.
    """
    # Flatten input
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = dip_freqs.flatten()
    sigma_flat = dip_errs.flatten()

    # Initial guesses (as before)
    idx_max = np.argmax(z_flat)
    x0_guess = x_flat[idx_max]
    y0_guess = y_flat[idx_max]
    gx_guess = (np.max(x_flat) - np.min(x_flat)) / 2
    gy_guess = (np.max(y_flat) - np.min(y_flat)) / 2
    baseline_guess = np.min(z_flat)
    A_guess = np.max(z_flat) - baseline_guess
    B_guess = baseline_guess

    p0 = [x0_guess, y0_guess, gx_guess, gy_guess, A_guess, B_guess]

    try:
        popt, pcov = curve_fit(gaussian_2d,
                               (x_flat, y_flat), z_flat,
                               p0=p0, sigma=sigma_flat, absolute_sigma=True)
        x0, y0, gx, gy, A, B = popt
        # Maximum value of the fitted surface:
        f_max = A + B
        # Error propagation from A and B
        var_A = pcov[4, 4]
        var_B = pcov[5, 5]
        cov_AB = pcov[4, 5]   # covariance between A and B
        f_max_err = np.sqrt(var_B + var_A + 2 * cov_AB)
        return f_max, f_max_err, (x0, y0), popt
    except (RuntimeError, ValueError):
        return np.nan, np.nan, (np.nan, np.nan)


# ------------------------------------------------------------
# 3. Running the full analysis pipeline
# ------------------------------------------------------------
# ============================================================
# 3.1 S Pole Analysis
# ============================================================
# Manually setting experiment parameters 
start_freq = 2.4e9   # 2.4 GHz
stop_freq  = 3.0e9   # 3.0 GHz
n_points = 151
# Generate frequency array manually since not reading from files
manual_freqs = np.linspace(start_freq, stop_freq, n_points)

# Path to the directory containing the experiment folders
data_root = 'C:\\School Code\\Magnet Scan S'   # <-- CHANGE THIS

# Read all experiments
all_data = read_all_experiments(data_root, manual_freqs=manual_freqs)

# ============================================================
# Process each experiment in the all_data dictionary
# ============================================================
#Prepair Empty Arrays to store results for all experiments
dip_freqs_list = []
dip_errs_list = []
dip_field_list = []

n=0

for timestamp, exp in all_data.items():
    n = n+1
    positions = exp['positions']          # shape (N, 2)
    values = exp['values']                # shape (N, n_freqs)
    freqs = exp['frequencies']            # 1D array of frequencies

    N_positions = positions.shape[0]
    dip_freqs = np.zeros(N_positions)
    dip_errs = np.zeros(N_positions)

    # Fit each position
    for i in range(N_positions):
        f0, err = fit_dip_per_position(freqs, values[i, :])
        dip_freqs[i] = f0
        dip_errs[i] = err

    # # Determine grid dimensions
    unique_x = np.unique(positions[:, 0])
    unique_y = np.unique(positions[:, 1])
    nx = len(unique_x)
    ny = len(unique_y)

    # # Sort and reshape to grid (same as before)
    dip_freqs_grid = dip_freqs.reshape(ny, nx)
    dip_errs_grid = dip_errs.reshape(ny, nx)

    # # Create coordinate grids
    Y_grid, X_grid = np.meshgrid(unique_x, unique_y)
    # 2D 2D fit for global maximum frequency
    f_max, f_max_err, (x0, y0), popt = fit_global_maximum_freq(X_grid, Y_grid,
                                                         dip_freqs_grid, dip_errs_grid)

    dip_freqs_list = np.append(dip_freqs_list, f_max)
    dip_errs_list = np.append(dip_errs_list, f_max_err)
    dip_field_list = np.append(dip_field_list, exp["B_surface"])

    gaussian_ref = gaussian_2d((X_grid, Y_grid), *popt).reshape(ny, nx)




    if n == 1:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X_grid/8*5, Y_grid/8*5, dip_freqs_grid/1e9, cmap='viridis', label='Sample Scan')
        ax.scatter(X_grid/8*5, Y_grid/8*5, gaussian_ref/1e9, cmap='inferno', label='Gaussian Fit', marker='o')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_zlabel('Resonance Frequency (GHz)')
        ax.set_title('Calibration Sample scan')
        ax.legend()

        plt.show()



dip_freqs_list = dip_freqs_list/1e9  # convert to GHz
dip_errs_list = dip_errs_list/1e9    # convert to GHz

# Perform linear regression
result_S = linregress(dip_field_list, dip_freqs_list)
print(f"Slope: {result_S.slope}, Intercept: {result_S.intercept}, R-squared: {result_S.rvalue**2}")
result_S_x = np.arange(400, 1500, 10)
result_S_y = result_S.slope * result_S_x + result_S.intercept



# ============================================================
# 3.2 N Pole Analysis
# ============================================================
# Manually setting experiment parameters
start_freq_N = 2.0e9   # 2.0 GHz
stop_freq_N = 2.6e9   # 2.6 GHz
n_points = 151
# Generate frequency array manually since not reading from files
manual_freqs = np.linspace(start_freq_N, stop_freq_N, n_points)

# Path to the directory containing the experiment folders
data_root = 'C:\\School Code\\Magnet Scan N'   # <-- CHANGE THIS

# Read all experiments
all_data_N = read_all_experiments(data_root, manual_freqs=manual_freqs)

# ============================================================
# Process each experiment in the all_data dictionary
# ============================================================

#Prepair Empty Arrays to store results for all experiments
dip_freqs_list_N = []
dip_errs_list_N = []
dip_field_list_N = []

for timestamp, exp in all_data_N.items():
    positions = exp['positions']          # shape (N, 2)
    values = exp['values']                # shape (N, n_freqs)
    freqs = exp['frequencies']            # 1D array of frequencies

    N_positions = positions.shape[0]
    dip_freqs = np.zeros(N_positions)
    dip_errs = np.zeros(N_positions)

    # 1D lorentzian fit for measurement at each position
    for i in range(N_positions):
        f0, err = fit_dip_per_position(freqs, values[i, :])
        dip_freqs[i] = f0
        dip_errs[i] = err

    # # Determine grid dimensions
    unique_x = np.unique(positions[:, 0])
    unique_y = np.unique(positions[:, 1])
    nx = len(unique_x)
    ny = len(unique_y)

    # # Sort and reshape to grid (same as before)
    dip_freqs_grid = dip_freqs.reshape(ny, nx)
    dip_errs_grid = dip_errs.reshape(ny, nx)

    # # Create coordinate grids
    Y_grid, X_grid = np.meshgrid(unique_x, unique_y)
    # # 2D fit for global minimum frequency
    f_min, f_min_err, (x0, y0), popt = fit_global_minimum_freq(X_grid, Y_grid,
                                                         dip_freqs_grid, dip_errs_grid)

    dip_freqs_list_N = np.append(dip_freqs_list_N, f_min)
    dip_errs_list_N = np.append(dip_errs_list_N, f_min_err)
    dip_field_list_N = np.append(dip_field_list_N, exp["B_surface"])

dip_freqs_list_N = dip_freqs_list_N/1e9  # convert to GHz
dip_errs_list_N = dip_errs_list_N/1e9    # convert to GHz

# Perform linear regression
result_N = linregress(dip_field_list_N, dip_freqs_list_N)
print(f"Slope: {result_N.slope}, Intercept: {result_N.intercept}, R-squared: {result_N.rvalue**2}")
result_N_x = np.arange(400, 1500, 10)
result_N_y = result_N.slope * result_N_x + result_N.intercept


# ============================================================
# 3.3 Combined Pole Analysis
# ============================================================
#Combining the S and N pole data for a total analysis
dip_field_list_N_inverted = [-x for x in dip_field_list_N]
dip_freqs_list_total = np.concatenate([dip_freqs_list, dip_freqs_list_N])
dip_errs_list_total = np.concatenate([dip_errs_list, dip_errs_list_N])
dip_field_list_total = np.concatenate([dip_field_list, dip_field_list_N_inverted])
# Perform linear regression
result_total = linregress(dip_field_list_total, dip_freqs_list_total)
print(f"Slope: {result_total.slope}, Intercept: {result_total.intercept}, R-squared: {result_total.rvalue**2}, Slope Error: {result_total.stderr}")
result_total_x = np.arange(-1500, 1500, 10)
result_total_y = result_total.slope * result_total_x + result_total.intercept

#calculate t value for slope
t_value = result_total.slope / result_total.stderr
print(f"T-value for slope: {t_value}")

#calculate p value for slope
from scipy.stats import t
degrees_of_freedom = len(dip_field_list_total) - 2  # n - 2 for linear regression
p_value = 2 * (1 - t.cdf(abs(t_value), df=degrees_of_freedom))
print(f"P-value for slope: {p_value}")


# ------------------------------------------------------------
# 4. Plotting results
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(result_S_x, result_S_y, '-', color='r', label=f'Fit: y={result_S.slope:.6f}x + {result_S.intercept:.3f}\n$R^2$={result_S.rvalue**2:.4f}')
plt.errorbar(dip_field_list, dip_freqs_list, yerr=dip_errs_list, fmt='.')
plt.xlabel('Surface Field Strength (Gs)')
plt.ylabel('Global Maximum Frequency (GHz)')
plt.title('Global Maximum Frequency vs Surface Field Strength(S)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(result_N_x, result_N_y, '-', color='r', label=f'Fit: y={result_N.slope:.6f}x + {result_N.intercept:.3f}\n$R^2$={result_N.rvalue**2:.4f}')
plt.errorbar(dip_field_list_N, dip_freqs_list_N, yerr=dip_errs_list_N, fmt='.')
plt.xlabel('Surface Field Strength (Gs)')
plt.ylabel('Global Maximum Frequency (GHz)')
plt.title('Global Maximum Frequency vs Surface Field Strength(N)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(result_total_x, result_total_y, '-', color='r', label=f'Fit: y={result_total.slope:.6f}x + {result_total.intercept:.3f}\n$R^2$={result_total.rvalue**2:.4f}\nSlope Error: +-{result_total.stderr:.6f} (1STD)\nintercept Error: +-{result_total.intercept_stderr:.3f} (1STD)')
plt.errorbar(dip_field_list_total, dip_freqs_list_total, yerr=dip_errs_list_total, fmt='.')

#line connecting the background point to the fit line at y=2.535 GHz
background_x = 0
background_y = 2.535
plt.plot(background_x, background_y, '.', color='b', markersize=10, label='Background Resonance Frequency=2.535 GHz')
bias_field = (background_y-result_total.intercept)/result_total.slope
plt.plot([background_x, bias_field], [background_y, background_y], '--', color='b', label=f'Bias Field: {bias_field:.1f} Gs')



plt.xlabel('Surface Field (Gs)')
plt.ylabel('Resonance Frequency (GHz)')
plt.title('Resonance Frequency - Surface Field Strength(Total)')
plt.grid(True)
plt.legend()
plt.show()