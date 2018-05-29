import logging

from distributed_vec_env import ClientConfiguration, EnvClient
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


class DeepMindEnv(EnvClient):
    def __init__(self, configuration: ClientConfiguration):
        super().__init__(configuration)

    def instantiate_env(self, environment_name, seed):
        """ Return a wrapped deepmind environment """
        env = make_atari(environment_name)
        env.seed(seed)
        return wrap_deepmind(env)


def main():
    """ Example program using EnvClient to run the environment """
    logging.basicConfig(level=logging.DEBUG)
    configuration = ClientConfiguration(
        server_url="tcp://localhost",
        command_port=9991,
        request_port=9992,
        server_version=1,
        timeout=10
    )

    client = DeepMindEnv(configuration)
    client.run()


if __name__ == '__main__':
    main()
