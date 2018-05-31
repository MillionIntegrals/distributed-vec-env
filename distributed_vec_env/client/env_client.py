import abc
import pickle

from gym import Env

from .client_base import ClientBase
from .client_configuration import ClientConfiguration

import distributed_vec_env.messages.protocol_pb2 as pb
import distributed_vec_env.numpy_util as numpy_util


class EnvClient(ClientBase):
    """ Client wrapping OpenAI gym environment """
    def __init__(self, configuration: ClientConfiguration):
        super().__init__(configuration)
        self.environment: Env = None

    ####################################################################################################################
    # Env interface
    @abc.abstractmethod
    def instantiate_env(self, environment_name, seed):
        """ Instantiate the environment : returns new environment """
        pass

    def initialize_env(self, environment_name, seed):
        """ Initialize internal environment """
        self.environment = self.instantiate_env(environment_name, seed)

    def env_space_payload(self):
        """ Populate environment initialization request """
        spaces = self.environment.observation_space, self.environment.action_space
        spaces_bytes = pickle.dumps(spaces)

        return pb.ConnectRequest(spaces=spaces_bytes)

    def close_env(self):
        """ Close the environment and free the resources """
        self.environment.close()

    def reset_env(self):
        """ Reset the environment and return next frame as Frame protocol buffer """
        observation = self.environment.reset()

        return pb.Frame(
            observation=numpy_util.serialize_numpy(observation),
            reward=0.0,
            done=False
        )

    def step_env(self, action):
        """ Perform action in the env and return next frame as Frame protocol buffer. """
        observation, reward, done, info = self.environment.step(action)

        return pb.Frame(
            observation=numpy_util.serialize_numpy(observation),
            reward=reward,
            done=done
        )
