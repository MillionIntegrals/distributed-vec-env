import zmq
import logging
import abc

import distributed_vec_env.messages.protocol_pb2 as pb

from .client_configuration import ClientConfiguration


class ClientInitializationException(Exception):
    """ Exception during client initialization """
    pass


class ClientCommandException(Exception):
    """ Exception during client initialization """
    pass


class ClientBase(abc.ABC):
    """
    This base class contains a stub implementation of DistributedVecEnv client.
    Ideally you should only need to override the env creation function
    """

    def __init__(self, configuration: ClientConfiguration):
        self.configuration = configuration
        self.context = zmq.Context()
        self.logger = logging.getLogger(__name__)

        self.command_socket = self.context.socket(zmq.SUB)
        self.command_socket.connect('{}:{}'.format(self.configuration.server_url, self.configuration.command_port))
        self.command_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.request_socket = self.context.socket(zmq.REQ)
        self.request_socket.connect('{}:{}'.format(self.configuration.server_url, self.configuration.request_port))

        # Received via initialization
        self.client_id = None
        self.environment_id = None
        self.server_config = None

    def run(self):
        """ Run this client """
        self.init()

        done = False

        while not done:
            command = self._fetch_command()
            done = self.run_command(command)

    ####################################################################################################################
    # Env interface
    @abc.abstractmethod
    def initialize_env(self, environment_name, seed):
        """ Initialize internal environment """
        pass

    @abc.abstractmethod
    def env_space_payload(self):
        """ Populate environment connection request """
        pass

    @abc.abstractmethod
    def close_env(self):
        """ Close the environment and free the resources """
        pass

    @abc.abstractmethod
    def reset_env(self):
        """ Reset the environment and return next frame as Frame protocol buffer """
        pass

    @abc.abstractmethod
    def step_env(self, action):
        """ Perform action in the env and return next frame as Frame protocol buffer. """
        pass

    ####################################################################################################################
    # Internal logic
    def init(self):
        """ Perform the initial handshake dance between the client and server to register for a mutual cooperation """
        self.logger.info(f"Worker uninitialized: waiting for environment name")
        environment_name, seed, self.client_id = self._send_initialize_request()
        self.initialize_env(environment_name, seed)

        self.logger.info(f"Worker {self.client_id}: waiting for registration")
        self.environment_id = self._send_connect_request(self.env_space_payload())
        self.logger.info(f"Worker {self.client_id}/{self.environment_id}: properly initialized")

    def run_command(self, message):
        """ Respond to a command received from the server """
        if message.command == pb.WorkerCommand.STEP:
            self.logger.info(f"Worker {self.client_id} received command STEP")
            self._send_frame(self.step_env(message.actions[self.environment_id]))
            return False
        elif message.command == pb.WorkerCommand.RESET:
            self.logger.info(f"Worker {self.client_id} received command RESET")
            self._send_frame(self.reset_env())
            return False
        elif message.command == pb.WorkerCommand.CLOSE:
            self.logger.info(f"Worker {self.client_id} received command CLOSE")
            self.close()
            return True
        else:
            raise ClientCommandException(f"Unknown command received from the server: {message}")

    def close(self):
        """ Close the client and free the resources """
        self.close_env()
        self.command_socket.close()
        self.request_socket.close()

    ####################################################################################################################
    # Requests to the server
    def _send_initialize_request(self):
        """ Request a name from the server """
        request = pb.MasterRequest(command=pb.MasterRequest.INITIALIZE)
        self.request_socket.send(request.SerializeToString())

        response = pb.NameResponse()
        response.ParseFromString(self.request_socket.recv())

        if response.server_version != self.configuration.server_version:
            raise ClientInitializationException(
                f"Server version {response.server_version} does not match client version "
                f"{self.configuration.server_version}"
            )

        return response.name, response.seed, response.client_id

    def _send_connect_request(self, payload):
        """ Register environment with the server """
        request = pb.MasterRequest(
            command=pb.MasterRequest.CONNECT,
            client_id=self.client_id,
            connect_payload=payload
        )
        self.request_socket.send(request.SerializeToString())

        response = pb.ConnectResponse()
        response.ParseFromString(self.request_socket.recv())

        return response.environment_id

    def _send_frame(self, frame):
        """ Send environment frame to the server """
        request = pb.MasterRequest(
            command=pb.MasterRequest.FRAME,
            client_id=self.client_id,
            frame=frame
        )
        self.request_socket.send(request.SerializeToString())
        self.request_socket.recv()  # Ignore the confirmation

    def _fetch_command(self):
        """ Wait for command from the server """
        message = pb.WorkerCommand()
        message.ParseFromString(self.command_socket.recv())
        return message

