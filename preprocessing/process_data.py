"""Load and save an array to shared memory."""
import argparse
import ctypes
import os.path
import sys

import numpy as np
import multiprocessing as mp

from functools import reduce

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to the data file.")
    parser.add_argument(
        "--name",
        help="File name to save in SharedArray. Defaults to the original file name.",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix to the file name to save in SharedArray. Only effective when "
        "`name` is not given.",
    )
    parser.add_argument(
        "--dtype", default="bool", help="Datatype of the array. Defaults to bool."
    )
    args = parser.parse_args()
    return args.filepath, args.name, args.prefix, args.dtype


def create_shared_array(shape, dtype):
    """Create shared array. Prompt if a file with the same name existed."""

    size=int(reduce((lambda x, y: x * y), list(shape)))
    shared_array=mp.Array(ctypes.c_bool,size)
    shared_array = np.ctypeslib.as_array(shared_array.get_obj())
    shared_array=shared_array.reshape(shape)
    return shared_array



def main():
    """Load and save an array to shared memory."""
    filepath, name, prefix, dtype = parse_arguments()

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + "_" + name

    print("Loading data from '{}'.".format(filepath))
    if filepath.endswith(".npy"):
        data = np.load(filepath)
        data = data.astype(dtype)
        sa_array = create_shared_array( data.shape, data.dtype)
        print("Saving data to shared memory...")
        np.copyto(sa_array, data)
    else:
        with np.load(filepath) as loaded:
            sa_array = create_shared_array(loaded["shape"], dtype)
            print("Saving data to shared memory...")
            sa_array[[x for x in loaded["nonzero"]]] = 1


    sa_array = sa_array.transpose(0, 1, 4, 2, 3)
    np.savez(name + ".npz", sa_array)
    print(
        "Successfully saved: (name='{}', shape={}, dtype={})".format(
            name, sa_array.shape, sa_array.dtype
        )
    )


if __name__ == "__main__":
    main()