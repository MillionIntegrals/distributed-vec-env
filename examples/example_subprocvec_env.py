from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def env_instantiate_fn(environment_name, seed):
    env = make_atari(environment_name)
    env.seed(seed)
    return wrap_deepmind(env)


def main():
    """
    Example program using SubProcVecEnv
    """
    num_envs = 2
    env_name = 'BreakoutNoFrameskip-v4'

    env = SubprocVecEnv([lambda: env_instantiate_fn(env_name, seed) for seed in range(num_envs)])
    obs = env.reset()

    print("After reset:")
    print(obs.shape)

    obs, rews, dones, infos = env.step([0, 0])

    print("After first action:")
    print(obs.shape)
    print(rews)
    print(dones)
    print(infos)

    obs, rews, dones, infos = env.step([1, 0])

    print("After second action:")
    print(obs.shape)
    print(rews)
    print(dones)
    print(infos)

    obs, rews, dones, infos = env.step([0, 1])

    print("After third action:")
    print(obs.shape)
    print(rews)
    print(dones)
    print(infos)

    env.close()


if __name__ == '__main__':
    main()
