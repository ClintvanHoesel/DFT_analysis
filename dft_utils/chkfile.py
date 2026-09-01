import sys

import h5py

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range


def load(chkfile, key):
    """Load array(s) from chkfile

    Args:
        chkfile : str
            Name of chkfile. The chkfile needs to be saved in HDF5 format.
        key : str
            HDF5.dataset name or group name.  If key is the name of a HDF5
            group, the group will be loaded into a Python dict, recursively.

    Returns:
        whatever read from chkfile
    """

    def load_as_dic(key, group):
        """Process load as dic."""
        if key in group:
            val = group[key]
        elif key + "__from_list__" in group:
            key = key + "__from_list__"
            val = group[key]
        else:
            return None

        if isinstance(val, h5py.Group):
            if key.endswith("__from_list__"):
                return [load_as_dic(k, val) for k in val]
            else:
                return {
                    k.replace("__from_list__", ""): load_as_dic(k, val) for k in val
                }
        else:
            return val[()]

    with h5py.File(chkfile, "r") as fh5:
        return load_as_dic(key, fh5)


load_chkfile_key = load


def dump(chkfile, key, value):
    """Save array(s) in chkfile

    Args:
        chkfile : str
            Name of chkfile.
        key : str
            key to be used in h5py object. It can contain "/" to represent the
            path in the HDF5 storage structure.
        value : array, vector, list ... or dict
            If value is a python dict or list, the key/value of the dict will
            be saved recursively as the HDF5 group/dataset structure.

    Returns:
        No return value
    """
    print(f"Saving to {chkfile}: {key}")

    def save_as_group(key, value, root):
        """Process save as group."""
        if isinstance(value, dict):
            root1 = root.create_group(key)
            for k in value:
                if not isinstance(k, str) or isinstance(k, bytes):
                    try:
                        k = str(k)
                    except:
                        k = bytes(k)
                save_as_group(k, value[k], root1)
        elif isinstance(value, (tuple, list, RANGE_TYPE)):
            root1 = root.create_group(key + "__from_list__")
            for k, v in enumerate(value):
                save_as_group("%06d" % k, v, root1)
        else:
            try:
                root[key] = value
            except (TypeError, ValueError) as e:
                if not (
                    e.args[0] == "Object dtype dtype('O') has no native HDF5 equivalent"
                    or e.args[0].startswith("could not broadcast input array")
                ):
                    raise e
                root1 = root.create_group(key + "__from_list__")
                for k, v in enumerate(value):
                    save_as_group("%06d" % k, v, root1)

    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile, "r+") as fh5:
            if key in fh5:
                del fh5[key]
            elif key + "__from_list__" in fh5:
                del fh5[key + "__from_list__"]
            save_as_group(key, value, fh5)
    else:
        with h5py.File(chkfile, "w") as fh5:
            save_as_group(key, value, fh5)
    print(f"Saved {chkfile}: {key}")


dump_chkfile_key = save = dump
