import tensorflow as tf
import tables
import os
import zarr
import numpy as np


def create_np_memmap_file(path, column_size, row_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.memmap(path, dtype='float32', mode='w+', shape=(column_size, row_size))


def update_column_to_memmap_file(path_with_name, data, columnumber, column_size, row_size):
    data = data.ravel()
    mem = np.memmap(path_with_name, dtype='float32', mode='r+', shape=(column_size, row_size))
    mem[:, columnumber] = data
    mem.flush()
    # z.append(data, axis=1)
    print(columnumber)


def create_zarr_file(path, column_size, row_size):  # zarr by far to slow
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # zarr.open(path, mode='w', shape=(column_size, 0), chunks = (column_size//row_size, row_size), dtype = 'f4',compressor=None)
    zarr.open(path, mode='w', shape=(0, column_size),
              chunks=(int(np.sqrt(row_size)), int(column_size // np.sqrt(row_size))), dtype='f4', compressor=None)

    # a=2

    # f = tables.open_file(path, mode='w')
    # atom = tables.Float32Atom()

    # array_c = f.create_earray(f.root, 'data', atom, (0, row_size)) # model params
    # f.create_earray(f.root, 'data', atom, (0, row_size)) # model params
    #
    # for idx in range(NUM_COLUMNS):
    #     x = np.random.rand(1, ROW_SIZE)
    #     array_c.append(x)
    # f.close()


def update_column_to_zarr_file(path_with_name, data, columnumber):
    # data=data.ravel()
    data = np.reshape(data, (1, -1))
    z = zarr.open(path_with_name, mode='r+')
    # z[:,columnumber]=data
    z.append(data, axis=0)
    print(columnumber)
    # a=2


def read_from_zarr_file(path_with_name, xfrom, xto, yfrom, yto):
    z = zarr.open(path_with_name, mode='r+')
    # data=f.root.data[1:10, 2:20]
    data = z[xfrom:xto, yfrom:yto]

    return data


def create_hdf5_file(path, row_size, column_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = tables.open_file(path, mode='w')
    atom = tables.Float32Atom()

    # array_c = f.create_earray(f.root, 'data', atom, (0, row_size)) # model params
    f.create_earray(f.root, 'data', atom, (0, row_size), expectedrows=column_size,
                    chunkshape=(int(np.sqrt(column_size)), row_size // np.sqrt(column_size)))  # model params
    # f.create_earray(f.root, 'data', atom, (0, row_size),expectedrows=column_size,chunkshape=(column_size,row_size//column_size)) # model params
    # 223
    # for idx in range(NUM_COLUMNS):
    #     x = np.random.rand(1, ROW_SIZE)
    #     array_c.append(x)
    f.close()


def append_to_hdf5_file(path_with_name, data):
    data = np.reshape(data, (1, -1))
    f = tables.open_file(path_with_name, mode='a')
    f.root.data.append(data)
    f.close()

    # print("save")
    # chunkshape


def read_from_hdf5_file(path_with_name, xfrom, xto, yfrom, yto):
    f = tables.open_file(path_with_name, mode='r')
    # data=f.root.data[1:10, 2:20]
    data = f.root.data[xfrom:xto, yfrom:yto]
    f.close()
    return data
