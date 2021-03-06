import logging
import hashlib
import tqdm

from distributed_vec_env import DistributedVecEnv
from distributed_vec_env import ServerConfiguration


def main():
    """
    Example program using DistributedVecEnv
    """
    logging.basicConfig(level=logging.DEBUG)

    configuration = ServerConfiguration(
        server_url="tcp://*",
        command_port=9991,
        request_port=9992,
        number_of_environments=2,
        environment_name='BreakoutNoFrameskip-v4',
        server_version=1,
        timeout=20,
        reset_compensation=True,
        verbosity=2
    )

    env = DistributedVecEnv(configuration)
    obs = env.reset()

    print("After reset:")
    print(obs.shape)
    for i in range(obs.shape[0]):
        print(f'Digest[{i}]', hashlib.sha256(obs[i].tobytes()).hexdigest())

    obs, rews, dones, infos = env.step([0, 0])

    print("After first action:")
    print(obs.shape)
    for i in range(obs.shape[0]):
        print(f'Digest[{i}]', hashlib.sha256(obs[i].tobytes()).hexdigest())
    print(rews)
    print(dones)
    print(infos)

    # while True:
    for i in tqdm.trange(10000):
        obs, rews, dones, infos = env.step([1, 0])

        # print("After second action:")
        # print(obs.shape)
        # for i in range(obs.shape[0]):
        #     print(f'Digest[{i}]', hashlib.sha256(obs[i].tobytes()).hexdigest())
        # print(rews)
        # print(dones)
        # print(infos)

    obs, rews, dones, infos = env.step([0, 1])

    print("After third action:")
    print(obs.shape)
    for i in range(obs.shape[0]):
        print(f'Digest[{i}]', hashlib.sha256(obs[i].tobytes()).hexdigest())
    print(rews)
    print(dones)
    print(infos)

    env.close()


if __name__ == '__main__':
    main()
