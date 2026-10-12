"""
Merge robomimic-style hdf5 files (e.g. scripted-demo shards) into one,
renumbering demos in input order.

Usage: python -m transfer.merge_hdf5 OUT.hdf5 IN1.hdf5 IN2.hdf5 ...
"""

import sys

import h5py


def main(out_path, in_paths):
    n = 0
    with h5py.File(out_path, 'w') as out:
        data = out.create_group('data')
        for path in in_paths:
            with h5py.File(path, 'r') as f:
                if 'env_args' not in data.attrs:
                    data.attrs['env_args'] = f['data'].attrs['env_args']
                for k in sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1])):
                    f.copy(f['data'][k], data, name=f'demo_{n}')
                    n += 1
    print(f"{out_path}: {n} demos from {len(in_paths)} files")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
