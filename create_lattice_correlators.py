import numpy as np
import os
import matplotlib.pyplot as plt


def read_files(file_path, Nt, correlators_per_measurement):
    """
    Passes directory to read_2pt_function and split_to_correlators functions.
    """
    print(f"Reading files from directory: {file_path}")
    files = sorted([f for f in os.listdir(file_path) if f.endswith('.meas')])
    print(f"Found {len(files)} files to process.")
    correlator_data = []
    for f in files:
        file_full_path = os.path.join(file_path, f)
        z_direction, x_direction = split_to_correlators(read_2pt_function(file_full_path), Nt, correlators_per_measurement)
        if len(z_direction) > 1:
            print("Multiple measurements in a single file detected!")
            measurements = len(z_direction)
            for i in range(measurements):
                z_direction_u, z_direction_d = split_u_d(z_direction[i])
                x_direction_u, x_direction_d = split_u_d(x_direction[i])
                data_dict = {
                    'z_u': z_direction_u,
                    'z_d': z_direction_d,
                    'x_u': x_direction_u,
                    'x_d': x_direction_d
                    }
                correlator_data.append(data_dict)
        else:
            z_direction_u, z_direction_d = split_u_d(z_direction)
            x_direction_u, x_direction_d = split_u_d(x_direction)
            data_dict = {
                'z_u': z_direction_u,
                'z_d': z_direction_d,
                'x_u': x_direction_u,
                'x_d': x_direction_d
                }
            correlator_data.append(data_dict)
    return correlator_data

def read_2pt_function(file_path):
    """
    Reads 2-point function data from a specified file path.

    Parameters:
    file_path (str): The path to the file containing 2-point function data.

    Returns:
    np.ndarray: A numpy array containing the 2-point function data.
    """
    raw_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('* 2pt'):
                line_content = line.split()
                correlator_data = line_content[-2] # all correlator data is in the second to last column
                try:
                    raw_data.append(float(correlator_data))
                except ValueError:
                    print(f"Warning: Could not convert {correlator_data} to float. Skipping this entry.")
    return np.array(raw_data)

def split_to_correlators(data, Nt, correlators_per_measurement):
    """
    Splits the 2-point function data into separate correlators.
    The the entries are split between the z and x directions.
    """
    data = np.asarray(data, dtype=float)
    if Nt <= 1:
        raise ValueError("Nt must be greater than 1.")
    if len(data) % Nt != 0:
        raise ValueError("Data length is not divisible by Nt.")

    num_correlators = len(data) // Nt
    correlators = [data[i * Nt:(i + 1) * Nt]
                   for i in range(num_correlators)]

    measurements = num_correlators // correlators_per_measurement
    print("This many measurements in the file:", measurements)

    z_direction = []
    x_direction = []

    for i in range(int(measurements)):  ### maybe I need to remove one from the max range here, lets see
        split_index = (i+1)*correlators_per_measurement - correlators_per_measurement//2
        z_direction.append(correlators[i*correlators_per_measurement:split_index])
        x_direction.append(correlators[split_index:(i+1)*correlators_per_measurement])

    return z_direction, x_direction

def split_u_d(data):
    """
    Splits the 2-point function data into up and down quark contributions.
    """
    data = np.asarray(data, dtype=float)
    data = np.squeeze(data)

    if len(data) < 2:
        raise ValueError("Need at least 2 entries to split into u/d.")

    zeroB = data[0]
    rest = data[1:]

    if len(rest) % 2 != 0:
        raise ValueError("Data length is not even, cannot split into up and down contributions.")

    u_correlators = np.concatenate(([zeroB], rest[0::2]))
    d_correlators = rest[1::2]
    return u_correlators, d_correlators

def jackknife_bins(data, n_bins):
    """
    Splits the data into jackknife bins.
    
    Parameters:
    data (np.ndarray): The data to be split into jackknife bins.
    n_bins (int): The number of jackknife bins to create.

    Returns:
    list: A list of jackknife bins.
    """
    n_data = len(data)
    bin_size = n_data // n_bins

    bins = []
    for i in range(n_bins):
        bin_data = np.delete(data, slice(i * bin_size, (i + 1) * bin_size))
        bins.append(np.mean(bin_data))
    return bins

def jackknife_samples(data):
    """
    Generates jackknife samples from the given data.

    Parameters:
    data (np.ndarray): The data from which to generate jackknife samples.

    Returns:
    list: A list of jackknife samples.
    """
    n = len(data)
    samples = []
    for i in range(n):
        sample = np.delete(data, i)
        samples.append(np.mean(sample))
    return samples

def jackknife(data, n_bins):
    """
    Computes the jackknife errors for the given data.

    Parameters:
    data (np.ndarray): The data for which to compute the jackknife errors.
    n_bins (int): The number of jackknife bins to create.

    Returns:
    tuple: The jackknife mean, jackknife error, and leave-one-bin-out estimates.
    """
    print(f"Computing jackknife errors for data of length {len(data)} with {n_bins} bins.")
    binned_data = jackknife_bins(data, n_bins)
    samples = np.asarray(jackknife_samples(binned_data), dtype=float)
    jackknife_mean = np.mean(samples)
    jackknife_error = np.sqrt((n_bins - 1) / n_bins * np.sum((samples - jackknife_mean)**2))
    return jackknife_mean, jackknife_error, samples

def save_correlator_jackknife(output_dir, correlator_name, Bfield_index, tau_points, jackknife_means, jackknife_errors, jackknife_estimates):
    """
    Save one correlator's tau-dependent jackknife output to a file.

    Columns:
        1. tau point
        2. jackknife mean
        3. jackknife error
        4 onward. leave-one-bin-out jackknife estimates
    """
    os.makedirs(output_dir, exist_ok=True)
    tau_points = np.asarray(tau_points, dtype=int)
    jackknife_means = np.asarray(jackknife_means, dtype=float)
    jackknife_errors = np.asarray(jackknife_errors, dtype=float)
    jackknife_estimates = np.asarray(jackknife_estimates, dtype=float)

    if not (len(tau_points) == len(jackknife_means) == len(jackknife_errors)):
        raise ValueError("tau_points, jackknife_means, and jackknife_errors must have the same length.")
    if jackknife_estimates.ndim != 2 or jackknife_estimates.shape[0] != len(tau_points):
        raise ValueError("jackknife_estimates must have one row per tau point.")

    B_fields = [0,1,2,4,6,8,10,12]

    if correlator_name.endswith('_d'):
        output_path = os.path.join(output_dir, f"emconduc_{len(tau_points)}_32_{correlator_name}_B{B_fields[Bfield_index+1]}.txt")    
    else:
        output_path = os.path.join(output_dir, f"emconduc_{len(tau_points)}_32_{correlator_name}_B{B_fields[Bfield_index]}.txt")
    table = np.column_stack((tau_points, jackknife_means, jackknife_errors, jackknife_estimates))
    header = ['tau', 'jackknife_mean', 'jackknife_error']
    header.extend(f'jackknife_bin_{i + 1}' for i in range(jackknife_estimates.shape[1]))
    np.savetxt(
        output_path,
        table,
        fmt=['%d', '%.12e', '%.12e'] + ['%.12e'] * jackknife_estimates.shape[1],
        header=' '.join(header),
        comments='# '
    )
    return output_path


def main():
    print("Starting analysis of lattice correlators...")
    Nt = 24
    file_path = f"/mnt/c/Users/chris/Desktop/lattice_data/quenched/finite_T/{Nt}x32/measurements/"
    correlators_per_measurement = 30
    correlator_data = read_files(file_path, Nt, correlators_per_measurement)
    output_dir = "/mnt/c/Users/chris/Desktop/pion_masses-main/jackknife_output/"

    to_analyze = ['z_u', 'z_d', 'x_u', 'x_d']

    for key_name in to_analyze:
        key_data = np.asarray([d[key_name] for d in correlator_data], dtype=float)
        print(f"{key_name} shape: {key_data.shape}")

        for Bfields in range(key_data.shape[1]):
            print(f"Analyzing {key_name} for B-field index {Bfields}")
            tau_points = np.arange(key_data.shape[-1])
            jackknife_means = []
            jackknife_errors = []
            jackknife_estimates = []
            for tau_point in tau_points:
                print(f"Analyzing {key_name} at tau point {tau_point}")
                jackknife_mean, jackknife_error, estimates = jackknife(key_data[:, Bfields, tau_point], n_bins=10)
                jackknife_means.append(jackknife_mean)
                jackknife_errors.append(jackknife_error)
                jackknife_estimates.append(estimates)

            B_fields = [0,1,2,4,6,8,10,12]
            plt.figure()
            plt.errorbar(tau_points, jackknife_means, yerr=jackknife_errors, fmt='x', markeredgewidth=1, elinewidth=1, capsize=3)
            plt.yscale('log')
            plt.xlabel('tau')
            plt.ylabel('Correlator')
            if key_name.endswith('_d'):
                plt.title(f'Correlator_{key_name}_{B_fields[Bfields+1]}')
                plt.savefig(os.path.join(output_dir, 'plots/', f'Correlator{Nt}_{key_name}_{B_fields[Bfields+1]}'))
            else:
                plt.title(f'Correlator_{key_name}_{B_fields[Bfields]}')
                plt.savefig(os.path.join(output_dir, 'plots/', f'Correlator{Nt}_{key_name}_{B_fields[Bfields]}'))
            plt.close()
            save_path = save_correlator_jackknife(output_dir, key_name, Bfields, tau_points, jackknife_means, jackknife_errors, jackknife_estimates)
            print(f"Saved {key_name} for B-field index {Bfields} results to {save_path}")


if __name__ == "__main__":
    main()
