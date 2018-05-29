import baselines.common.vec_env as v

import distributed_vec_env.server as server


class DistributedVecEnv(v.VecEnv):
    """
    Distributed version of Vector Environment.
    Send commands to a distributed set of nodes.
    Mostly just a very thin wrapper over ServerConnection
    """

    def __init__(self, configuration: server.ServerConfiguration):
        self.configuration = configuration
        self.connection = server.ServerConnection(self.configuration)

        observation_space, action_space = self.connection.initialize()

        super().__init__(
            num_envs=self.configuration.number_of_environments,
            observation_space=observation_space,
            action_space=action_space
        )

    def reset(self):
        """
        Reset all the environments and return an array of
        observations.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        return self.connection.reset_environments()

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        self.connection.send_actions(actions)

    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: an array of info objects
        """
        return self.connection.gather_frames()

    def close(self):
        """
        Clean up the environments' resources.
        """
        self.connection.close_environments()

