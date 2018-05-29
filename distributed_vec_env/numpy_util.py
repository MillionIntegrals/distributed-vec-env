import numpy as np

import distributed_vec_env.messages.protocol_pb2 as pb


def serialize_numpy(array):
    """ Serialize numpy array into a protocol buffer """

    return pb.Array(
        data=array.tobytes(),
        shape=list(array.shape),
        dtype=array.dtype.name
    )


def deserialize_numpy(protobuf):
    """ Deserialize array protobuf into a numpy array """
    buf = memoryview(protobuf.data)
    arr = np.frombuffer(buf, dtype=protobuf.dtype)
    reshaped = arr.reshape(tuple(protobuf.shape))

    return reshaped
