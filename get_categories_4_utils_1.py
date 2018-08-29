__doc__="""This script contains the functions to load and save data from and to disk"""


import pickle


def get_data(fname):
    """gets data from disk"""
    # You could also add "and i.startswith('f')
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
    handle.close()
    return data
    
                
def save_data(data, data_fname, ext='pickle'):
    """saves data to disk as pickle or text file"""
    if ext == 'pickle':
        with open(data_fname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_fname, 'w') as handle:
            handle.write(data)
