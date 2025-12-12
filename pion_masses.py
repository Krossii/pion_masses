import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os


home_dir = '/mnt/c/Users/chris/OneDrive/Desktop/'

n_correlators = 2
n_tau = 48
n_configs = 151

def read_lattice_data(file_path):
    """
    Reads Lattice QCD data from a given file.
    
    Parameters:
    file_path (str): Path to the data files.

    Returns:
    np.ndarray: Array of data read from the files.
    """
    raw_data = []
    inside = False
    with open(file_path, 'r') as f:
        for line in f:
            if 'Measured 2pt functions:' in line:
                inside = True
                continue
            if inside:
                toks = line.split()
                raw_data.append(toks)

    if not raw_data:
        raise ValueError("No data found in the file.")

    def try_float(s):
        try:
            return float(s)
        except ValueError:
            return None

    numeric_rows = []
    for row in raw_data:
        nums = [try_float(x) for x in row]
        nums = [x for x in nums if x is not None]
        if nums:
            numeric_rows.append(nums)

    del numeric_rows[-1] # remove last row where the deletion of the propagators is logged
    data = np.array(numeric_rows, dtype=float)
    data = data[:,6:-1] # only retain the correlator data
    return np.squeeze(data)

def create_bins(data, bin_size): ### check this guy
    """
    Creates binned data from the given data.
    
    Parameters:
    data (np.ndarray): Original data array.
    bin_size (int): Size of each bin.

    Returns:
    np.ndarray: Binned data array.
    """
    n_data = len(data)
    n_bins = n_data // bin_size
    binned_data = np.zeros((n_bins, data.shape[1]))
    for i in range(n_bins):
        binned_data[i] = np.mean(data[i*bin_size:(i+1)*bin_size], axis=0)
    return binned_data

def create_jackknife_samples(data, n_jackknife=None): ### check this guy
    """
    Creates jackknife samples from the given data.
    
    Parameters:
    data (np.ndarray): Original data array.
    n_jackknife (int): Number of jackknife samples to create.

    Returns:
    np.ndarray: Array of jackknife samples.
    """
    n_data = data.shape[0]
    if n_jackknife is None:
        n_jackknife = n_data
    # each jackknife sample is the mean over the remaining rows -> length = n_points
    n_points = data.shape[1]
    jackknife_samples = np.zeros((n_jackknife, n_points))
    for i in range(n_jackknife):
        mask = np.ones(n_data, dtype=bool) # create a boolean mask
        mask[i % n_data] = False # leave out the i-th data point
        jackknife_samples[i] = np.mean(data[mask], axis=0) # compute the mean of the remaining data points
    return jackknife_samples

def fit_function(tau, A, m):
    return A * (np.exp(-m * tau) + np.exp(-m * (n_tau - tau)))

def main():
    data = []
    for files in os.listdir(os.path.join(home_dir, 'pion_lattice_data')):
        data.append(read_lattice_data(os.path.join(home_dir, 'pion_lattice_data', files)))
    print("Data read from file:")
    data = np.array(data) # the data array is the concatenated correlators for each configuration
    correlators = np.zeros((n_correlators, n_configs, n_tau))
    for i in range(n_correlators):
        correlators[i] = data[:,i*n_tau:(i+1)*n_tau] # seperate the correlators
    print(correlators.shape)
    ### Create binned data for each correlator
    bin_size = 10
    binned_correlators = []
    for i in range(n_correlators):
        binned_data = create_bins(correlators[i], bin_size)
        binned_correlators.append(binned_data)
    binned_correlators = np.array(binned_correlators)
    print("Binned correlators shape:", binned_correlators.shape)
    ### Create jackknife samples for each correlator
    n_jackknife = binned_correlators.shape[1]
    jackknife_correlators = []
    for i in range(n_correlators):
        jackknife_data = create_jackknife_samples(binned_correlators[i], n_jackknife)
        jackknife_correlators.append(jackknife_data)
    jackknife_data = np.array(jackknife_data)
    print("Jackknife correlators shape:", jackknife_data.shape)
    ### Save the processed data
    np.save(os.path.join(home_dir, 'pion_masses/', 'binned_correlators.npy'), binned_correlators)
    np.save(os.path.join(home_dir, 'pion_masses/', 'jackknife_correlators.npy'), jackknife_correlators)
    print("Processed data saved.")
    ### Fit the correlators to extract masses 
    ### (note that the range of tau values for the fit function needs to be adjusted)
    masses = []
    for i in range(n_correlators):
        mean_correlator = np.mean(correlators[i], axis=0)
        popt, pcov = curve_fit(fit_function, np.arange(n_tau), mean_correlator, p0=(1.0, 0.1))
        masses.append(popt[1])
        print(f'Fitted mass for correlator {i+1}: {popt[1]}')
    ### Save the fitted masses
    np.save(os.path.join(home_dir, 'pion_masses/', 'fitted_masses.npy'), np.array(masses))
    print("Fitted masses saved.")
    ### Jackknife error estimation for masses (rethink if this is correct)
    ### (note that the range of tau values for the fit function needs to be adjusted)
    jackknife_masses = []
    for i in range(n_correlators):
        jk_masses = []
        for j in range(n_jackknife):
            jk_sample = jackknife_correlators[i][j]
            popt, pcov = curve_fit(fit_function, np.arange(n_tau), jk_sample, p0=(1.0, 0.1))
            jk_masses.append(popt[1])
        jk_masses = np.array(jk_masses)
        mean_mass = np.mean(jk_masses)
        error = np.sqrt((n_jackknife - 1) * np.mean((jk_masses - mean_mass)**2))
        jackknife_masses.append((mean_mass, error))
        print(f'Jackknife mass for correlator {i+1}: {mean_mass} ± {error}')
    ### Save the jackknife masses
    np.save(os.path.join(home_dir, 'pion_masses/', 'jackknife_masses.npy'), np.array(jackknife_masses))
    print("Jackknife masses saved.")
    ### Plot the correlators
    tau = np.arange(n_tau)
    for i in range(n_correlators):
        mean_correlator = np.mean(correlators[i], axis=0)
        plt.figure()
        plt.errorbar(tau, mean_correlator, yerr=np.std(correlators[i], axis=0), fmt='o', alpha=0.5)
        plt.plot(tau, fit_function(tau, *popt), 'r-', label='Fit')
        plt.yscale('log')
        plt.xlabel('tau')
        plt.ylabel('Correlator')
        plt.title(f'Correlator {i+1} vs tau')
        plt.legend()
        plt.savefig(os.path.join(home_dir, 'pion_masses/', f'correlator_{i+1}.png'))
        plt.close()


if __name__ == "__main__":
    main()