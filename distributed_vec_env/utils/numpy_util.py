import numpy as np


def deserialize_numpy(msg):
    buf = memoryview(msg.data)
    arr = np.frombuffer(buf, dtype=msg.dtype)
    reshaped = arr.reshape(tuple(msg.shape))

    return reshaped
