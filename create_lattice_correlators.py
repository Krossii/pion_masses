import numpy as np
import os

def read_files(file_path, correlators_per_file):
    """
    Passes directory to read_2pt_function and split_to_correlators functions.
    """
    print(f"Reading files from directory: {file_path}")
    files = sorted([f for f in os.listdir(file_path) if f.endswith('.meas')])
    print(f"Found {len(files)} files to process.")
    correlator_data = []
    for f in files:
        file_full_path = os.path.join(file_path, f)
        zero_B, z_direction, x_direction = split_to_correlators(read_2pt_function(file_full_path), correlators_per_file)
        z_direction_u, z_direction_d = split_u_d(z_direction)
        x_direction_u, x_direction_d = split_u_d(x_direction)
        data_dict = {
            'zero_B': zero_B,
            'z_direction_u': z_direction_u,
            'z_direction_d': z_direction_d,
            'x_direction_u': x_direction_u,
            'x_direction_d': x_direction_d
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

def split_to_correlators(data, num_correlators):
    """
    Splits the 2-point function data into separate correlators.
    The first entry is the zero-B correlator; the remaining entries are split
    between the z and x directions.
    """
    data = np.asarray(data, dtype=float)
    if num_correlators <= 1:
        raise ValueError("num_correlators must be greater than 1.")
    if len(data) % num_correlators != 0:
        raise ValueError("Data length is not divisible by the number of correlators.")

    correlator_length = len(data) // num_correlators
    correlators = [data[i * correlator_length:(i + 1) * correlator_length]
                   for i in range(num_correlators)]

    zero_B_correlator = correlators[0]

    if num_correlators == 2:
        z_direction = correlators[1:]
        x_direction = np.array([], dtype=float)
    else:
        split_index = num_correlators// 2
        z_direction = correlators[1:split_index]
        x_direction = correlators[1 + split_index:]

    print(len(zero_B_correlator), len(z_direction), len(x_direction))

    return zero_B_correlator, z_direction, x_direction

def split_u_d(data):
    """
    Splits the 2-point function data into up and down quark contributions.
    """
    data = np.asarray(data, dtype=float)
    print(len(data))
    if len(data) % 2 != 0:
        raise ValueError("Data length is not even, cannot split into up and down contributions.")

    u_correlators = data[0::2]
    d_correlators = data[1::2]
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
        print(np.mean(bin_data)) ### this should print one number for each bin
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
    tuple: A tuple containing the jackknife mean and jackknife error for the data.
    """
    print(f"Computing jackknife errors for data of length {len(data)} with {n_bins} bins.")
    binned_data = jackknife_bins(data, n_bins)
    samples = jackknife_samples(binned_data)
    jackknife_mean = np.mean(samples)
    jackknife_error = np.sqrt((n_bins - 1) / n_bins * np.sum((samples - jackknife_mean)**2))
    return jackknife_mean, jackknife_error

def main():
    print("Starting analysis of lattice correlators...")
    file_path = "/home/Christian/Desktop/lattice_data/quenched/T0/96x32/"
    correlators_per_file = 30
    correlator_data = read_files(file_path, correlators_per_file)

    to_analyze = ['zero_B', 'z_direction_u', 'z_direction_d', 'x_direction_u', 'x_direction_d']

    for key_name in to_analyze:
        key_data = np.asarray([d[key_name] for d in correlator_data], dtype=float)
        print(f"{key_name} shape: {key_data.shape}")

        if key_data.ndim == 1:
            key_data = key_data[np.newaxis, :]

        for tau_point in range(key_data.shape[1]):
            print(f"Analyzing {key_name} at tau point {tau_point}")
            jackknife_mean, jackknife_error = jackknife(key_data[:, tau_point], n_bins=10)
            print(f"Jackknife mean at tau point {tau_point}: {jackknife_mean}")
            print(f"Jackknife error at tau point {tau_point}: {jackknife_error}")
            print()

    ### find a nice way to save the results to a file


if __name__ == "__main__":
    main()
