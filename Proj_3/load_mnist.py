import numpy as np
import gzip

# Using this as a reference for format
# https://yann.lecun.com/exdb/mnist/


def load_data_arr(path, render_d=True):
    f = gzip.open(path, "r")

    # First 2 bytes are magic number, 3rd encodes datatype
    datatype = f.read(3)
    datatype = int.from_bytes(datatype, byteorder="big")
    encoded_dtype = np.uint8
    bytes_per_dim = 1

    if datatype != 8:
        raise Exception("Unsupported Dtype")

    raw_dims = f.read(1)
    dims = int.from_bytes(raw_dims, byteorder="big")
    dimension_data = []
    num_bytes = 1
    for i in range(0, dims):
        current_dim_size = int.from_bytes(f.read(4), byteorder="big")
        dimension_data.append(current_dim_size)
        num_bytes = num_bytes * current_dim_size * bytes_per_dim

    buf = f.read(num_bytes)
    data = np.frombuffer(buf, dtype=encoded_dtype).reshape(dimension_data)
    return data


if __name__ == "__main__":
    data = load_data_arr("MNIST/train-labels-idx1-ubyte.gz")
    print(data)
