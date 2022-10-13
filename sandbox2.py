import h5py

if __name__ == '__main__':
    with h5py.File("data.hdf5", "r") as f:
        # We use [:] slice operator to create a copy that persists after we
        # close the file / exit the context manager
        X = f["Membrane Positions: X"][:]
        Y = f["Membrane Positions: Y"][:]
        p = f["Fluid Pressure Field"][:]
        t = f["t"][()]
