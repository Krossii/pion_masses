import numpy as np

def read_lattice_data(file_path):
    """
    Reads Lattice QCD data from a given file.
    
    Parameters:
    file_path (str): Path to the data files.

    Returns:
    np.ndarray: Array of data read from the files.
    """
    with open(file_path, 'r') as f:
        for line in f:
            if 'functions:' in line:
                for line in f:
                    data = np.array([float(x) for x in line.split()])
                    return data
    raise ValueError("No data found in the file.")

