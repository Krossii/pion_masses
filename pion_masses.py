import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import os


home_dir = '/mnt/c/Users/chris/OneDrive/Desktop/'

n_correlators = 8
n_tau = 48
n_configs = 50

# relevant variables for plotting
tau = np.arange(n_tau)
colors = ['b', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple']
kappa_values = np.array([0.1337, 0.1339, 0.1341, 0.1343, 0.1345, 0.1347, 0.1348, 0.13485]) # coming up: 0.13490

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

    #del numeric_rows[-1] # remove last row where the deletion of the propagators is logged
    data = np.array(numeric_rows, dtype=float)
    data = data[:,6:-1] # only retain the correlator data
    return np.squeeze(data)

def create_binned_data(correlators, bin_size=10):
    binned_correlators = []
    for i in range(n_correlators):
        binned_data = create_bins(correlators[i], bin_size)
        binned_correlators.append(binned_data)
    binned_correlators = np.array(binned_correlators)
    print("Binned correlators shape:", binned_correlators.shape)
    return binned_correlators

def create_bins(data, bin_size):     
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

def create_jackknife_data(binned_correlators, n_jackknife=None):
    jackknife_correlators = []
    for i in range(n_correlators):
        jackknife_data = create_jackknife_samples(binned_correlators[i], n_jackknife)
        jackknife_correlators.append(jackknife_data)
    jackknife_data = np.array(jackknife_correlators)
    jackknife_error = np.sqrt((n_jackknife - 1) * np.mean((jackknife_data - np.mean(jackknife_data, axis=1, keepdims=True))**2, axis=1, keepdims = True))
    print("Jackknife correlators shape:", jackknife_data.shape)
    return jackknife_data, np.squeeze(jackknife_error)

def create_jackknife_samples(data, n_jackknife=None):
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
    e1 = np.exp(np.clip(-m *tau, -700, 700))
    e2 = np.exp(np.clip(-m *(n_tau - tau), -700, 700))
    return A * (e1+e2)

def chi_squared(observed, expected, ndof, errors):
    r = observed - expected
    chi2 = np.sum((r / errors) ** 2)
    chi2 /= ndof
    return chi2

def lattice_spacing_sommer(am): # convert to fm
    r0 = 0.5 # fermi
    beta = 6.872
    a = r0 * np.exp(-1.6804 - 1.7331*(beta - 6) + 0.7849*(beta - 6)**2 - 0.4428*(beta - 6)**3)
    print(f'Lattice spacing a: {a} fm')
    r0_ratio = r0 / a
    print(f'r0/a: {r0_ratio}')
    rm = r0_ratio * am
    m_pi = rm / r0 *197.3269804 # in MeV
    return rm, m_pi

def lattice_spacing_olaf(am):    
    r0 = 0.5 # fermi
    m_div_T = 0.0243 # from paper
    m = 9.5 # MeV from paper
    a = m_div_T / (16 * m)* 197.3269804 # in fm
    print(f'Lattice spacing a: {a} fm')
    r0_ratio = r0 / a
    print(f'r0/a: {r0_ratio}')
    rm = r0_ratio * am
    m_pi = rm / r0 *197.3269804 # in MeV
    return rm, m_pi

def load_run(run_dir):
    """Load all files in run_dir and return array shape (n_configs_run, n_correlators_per_run, n_tau)."""
    files = sorted([f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))])
    run_data = [read_lattice_data(os.path.join(run_dir, f)) for f in files]
    run_data = np.vstack(run_data)  # (n_configs_run, n_correlators * n_tau)
    if run_data.shape[1]%n_tau != 0:
        raise ValueError(f"Got {run_data.shape[1]} columns, expected a multiple of {n_tau}")
    return run_data.reshape(run_data.shape[0], run_data.shape[1] // n_tau, n_tau)

def fit_correlator(jk_correlators, jk_error, fitrange):
    masses = []
    factor = []
    chisq = []
    jk_error = jk_error[:, int(n_tau/2)-fitrange: int(n_tau/2)+fitrange]
    for i in range(n_correlators):
        mean_correlator = np.mean(jk_correlators, axis=1)[i]
        mean_correlator = mean_correlator[int(n_tau/2)-fitrange: int(n_tau/2)+fitrange]
        popt, pcov = curve_fit(
            fit_function, 
            np.arange(int(n_tau/2)-fitrange,  int(n_tau/2)+fitrange), 
            mean_correlator, 
            sigma = jk_error[i][:])
        ndof = len(mean_correlator) - len(popt)
        factor.append(popt[0])
        masses.append(popt[1])
        chisq.append(chi_squared(mean_correlator, fit_function(np.arange(int(n_tau/2)-fitrange, int(n_tau/2)+fitrange), *popt), ndof, jk_error))
        print(f'Fitted mass for correlator {i+1}: {popt[1]}')
    return masses, factor, chisq

def jackknife_mass_errors(jk_correlators, jk_error, n_jackknife, fitrange):
    jackknife_masses = []
    jackknife_masses_errors = []
    jk_error = jk_error[:, int(n_tau/2)-fitrange: int(n_tau/2)+fitrange]
    for i in range(n_correlators):
        jk_masses = []
        jk_factors = []
        for j in range(n_jackknife):
            jk_sample = jk_correlators[i][j]
            jk_sample = jk_sample[int(n_tau/2)-fitrange: int(n_tau/2)+fitrange]
            popt, pcov = curve_fit(
                fit_function, 
                np.arange(int(n_tau/2)-fitrange,  int(n_tau/2)+fitrange), 
                jk_sample, 
                sigma = jk_error[i][:])
            jk_factors.append(popt[0])
            jk_masses.append(popt[1])
        jk_masses = np.array(jk_masses)
        mean_mass = np.mean(jk_masses)
        error = np.sqrt((n_jackknife - 1) * np.mean((jk_masses - mean_mass)**2))
        jackknife_masses_errors.append(error)
        jackknife_masses.append((mean_mass, error))
        print(f'Jackknife mass for correlator {i+1}: {mean_mass} ± {error}')
        return jackknife_masses, jackknife_masses_errors

def plot_correlators(correlators, jackknife_error, masses, factor, chisq, fitrange):
    fit_range = np.arange(int(n_tau/2)-fitrange,  int(n_tau/2)+fitrange)
    plt.figure()
    for i in range(n_correlators):
        #yerr = np.std(correlators[i], axis=0)
        mean_correlator = np.mean(correlators[i], axis=0)
        plt.errorbar(tau, mean_correlator, yerr=jackknife_error[i][:], fmt='x', markeredgewidth=1, elinewidth=1, capsize=3, alpha=0.4, color=colors[i])
        plt.plot(fit_range, fit_function(fit_range, factor[i], masses[i]), label=f'Chisq:{chisq[i]:.2f}', color=colors[i])
    plt.yscale('log')
    plt.xlabel('tau')
    plt.ylabel('Correlator')
    plt.title('Correlators vs tau')
    plt.legend()
    plt.savefig(os.path.join(home_dir, 'pion_masses/', 'correlators.png'))
    plt.close()

def plot_effective_mass(correlators):
    m_eff = np.zeros(n_tau-2)
    plt.figure()
    for i in range(n_correlators):
        mean_correlator = np.mean(correlators[i], axis=0)
        for j in range(1,n_tau-1):
            m_eff[j-1] = np.abs(1/2*np.log(mean_correlator[j+1]/mean_correlator[j-1]))
        plt.plot(tau[1:-1], m_eff, 'x-', label=f'Correlator {i+1}', color=colors[i])
    plt.xlabel('tau')
    plt.ylabel('abs(Effective Mass)')
    plt.title('abs(Effective Mass) vs tau')
    plt.savefig(os.path.join(home_dir, 'pion_masses/', 'effective_mass.png'))
    plt.close()

def fit_kappa_values(m_pi, jk_masses_errors):
    p = np.polyfit(1/kappa_values, m_pi**2, 1, w = 2*m_pi*np.array(jk_masses_errors))
    print(f'Fit parameters (slope, intercept): {p}')
    pion_mass = 400 # MeV
    determined_kappa = p[0] / (pion_mass**2 - p[1])
    print(f'Determined kappa for pion mass {pion_mass} MeV: {determined_kappa}')
    return p, determined_kappa

def plot_kappa_values(p, m_pi, pion_mass, determined_kappa, jk_masses_errors):
    paper_kappa = 0.13495
    physical_pion = 135**2 # MeV^2

    plt.figure()
    plt.errorbar(1/kappa_values, m_pi**2, yerr=2*m_pi*np.array(jk_masses_errors), fmt='x', markeredgewidth=1, elinewidth=1, capsize=3)
    plt.plot(1/kappa_values, np.polyval(p, 1/kappa_values), '-', label=f'Fit: y={p[0]:.2f}x + {p[1]:.2f}')
    plt.plot(1/determined_kappa, pion_mass**2, 'rx', label=f'Kappa for {pion_mass} MeV: {determined_kappa:.5f}')
    plt.vlines(1/paper_kappa, label = f'Olafs kappa: {paper_kappa}', ymin=0, ymax=pion_mass**2+1000000, colors='g', linestyles='dashed')
    plt.hlines(physical_pion, label = f'$m_p$ = {np.sqrt(physical_pion):.2f} MeV', xmin=7.38, xmax=7.43, colors='m', linestyles='dashed')
    plt.hlines(pion_mass**2, label = f'$m_p$ = {pion_mass} MeV', xmin=7.38, xmax=7.43, colors='r', linestyles='dashed')
    plt.legend()
    plt.xlabel('1/kappa')
    plt.ylabel('Pion Mass^2 (MeV^2)')
    plt.title('Pion Mass^2 vs 1/Kappa')
    plt.savefig(os.path.join(home_dir, 'pion_masses/', 'pion_mass_vs_kappa.png'))
    plt.close()


def main():
    run_paths = [
        os.path.join(home_dir, 'pion_lattice_data/second_run/'),
        os.path.join(home_dir, 'pion_lattice_data/third_run/'),
        os.path.join(home_dir, 'pion_lattice_data/fourth_run/'),
        os.path.join(home_dir, 'pion_lattice_data/fifth_run/'),
    ]

    runs = [load_run(p) for p in run_paths]               # each run: (n_configs_run, n_correlators, n_tau)
    
    all_runs = np.concatenate(runs, axis=1)               # (n_configs, n_correlators, n_tau)

    correlators = np.transpose(all_runs, (1, 0, 2))       # (n_correlator, n_configs_total, n_tau)

    print("Correlators shape:", correlators.shape)
    ### Create binned data for each correlator
    binned_correlators = create_binned_data(correlators, bin_size=10)
    ### Create jackknife samples for each correlator
    n_jackknife = binned_correlators.shape[1]
    jackknife_correlators, jackknife_error = create_jackknife_data(binned_correlators, n_jackknife)
    ### Save the processed data
    np.save(os.path.join(home_dir, 'pion_masses/', 'binned_correlators.npy'), binned_correlators)
    np.save(os.path.join(home_dir, 'pion_masses/', 'jackknife_correlators_error.npy'), jackknife_correlators, jackknife_error)
    print("Processed data saved.")
    ### Fit the correlators to extract masses 
    ### (note that the range of tau values for the fit function needs to be adjusted by effective mass)
    fitrange = 16
    masses, factor, chisq = fit_correlator(jackknife_correlators, jackknife_error, fitrange)    
    ### Save the fitted masses
    np.save(os.path.join(home_dir, 'pion_masses/', 'fitted_masses.npy'), np.array(masses))
    print("Fitted masses saved.")
    ### Jackknife error estimation for masses
    jackknife_masses, jackknife_masses_errors = jackknife_mass_errors(jackknife_correlators, jackknife_error, n_jackknife, fitrange)
    ### Save the jackknife masses
    np.save(os.path.join(home_dir, 'pion_masses/', 'jackknife_masses.npy'), np.array(jackknife_masses))
    print("Jackknife masses saved.")
    ### Plot the correlators
    plot_correlators(correlators, jackknife_error, masses, factor, chisq, fitrange)
    ### Plot the effective mass for the first correlator
    plot_effective_mass(correlators)

    ### Physical mass
    sommer_scale = False
    if sommer_scale: # from sommer scale
        rm, m_pi = lattice_spacing_sommer(np.array(masses))
        print(f'Physical mass cross check with r_0: {rm}') 
    else: # from Olafs paper
        rm, m_pi = lattice_spacing_olaf(np.array(masses))
        print(f'Physical mass cross check with r_0: {rm} MeV')

    print(f'Physical pion mass: {m_pi} MeV')

    fit, determined_kappa = fit_kappa_values(m_pi, jackknife_masses_errors)
    pion_mass = 400 # MeV
    plot_kappa_values(fit, m_pi, pion_mass, determined_kappa, jackknife_masses_errors)

if __name__ == "__main__":
    main()