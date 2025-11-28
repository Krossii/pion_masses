import numpy as np
import os


home_dir = '/mnt/c/Users/chris/OneDrive/Desktop/'

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
                #if not toks:  # stop on blank line (adjust if file uses different separator)
                #    break
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

def main():
    # could add automation for more files here
    # remember that the measurement ids are different for some files
    data_file = os.path.join(home_dir, 'pion_lattice_data', '48x48x48x48_SU3_b6.872000_id100_mid1.meas')
    data = read_lattice_data(data_file)
    print("Data read from file:")
    print(data) # could add a split function here to seperate the different correlators
    # right now, the data array is two concatenated correlators



if __name__ == "__main__":
    main()