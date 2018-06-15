import logging
import pickle
import time

import distributed_vec_env.messages.protocol_pb2 as pb
import distributed_vec_env.numpy_util as numpy_util

from distributed_vec_env import ClientConfiguration, EnvClient
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


CLIENT_RESET_SLEEP = 2


class DeepMindEnv(EnvClient):
    def __init__(self, configuration: ClientConfiguration):
        super().__init__(configuration)

    def instantiate_env(self, environment_name, seed):
        """ Return a wrapped deepmind environment """
        env = make_atari(environment_name)
        env.seed(seed)
        return wrap_deepmind(env)

    def reset_env(self):
        """ Reset the environment and return next frame as Frame protocol buffer """
        observation = self.environment.reset()
        time.sleep(CLIENT_RESET_SLEEP)  # SLEEP ON RESET

        return pb.Frame(
            observation=numpy_util.serialize_numpy(observation),
            reward=0.0,
            done=False,
            info=pickle.dumps({})
        )


def main():
    """ Example program using EnvClient to run the environment """
    logging.basicConfig(level=logging.DEBUG)
    configuration = ClientConfiguration(
        server_url="tcp://localhost",
        command_port=9991,
        request_port=9992,
        server_version=1,
        timeout=5,
        wait_period=1,
        polling_limit=5
    )

    client = DeepMindEnv(configuration)
    client.run()


if __name__ == '__main__':
    main()
