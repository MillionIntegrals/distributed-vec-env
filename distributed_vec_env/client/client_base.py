import abc
import pickle
import time
import zmq

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
        self.logger = self.configuration.logger

        self.command_socket = self.context.socket(zmq.SUB)
        self.command_socket.setsockopt(zmq.LINGER, self.configuration.linger_period * 1000)
        self.command_socket.connect('{}:{}'.format(self.configuration.server_url, self.configuration.command_port))
        self.command_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.command_poller = zmq.Poller()
        self.command_poller.register(self.command_socket, zmq.POLLIN)

        self.request_socket = self.context.socket(zmq.REQ)
        self.request_socket.setsockopt(zmq.LINGER, self.configuration.linger_period * 1000)
        self.request_socket.connect('{}:{}'.format(self.configuration.server_url, self.configuration.request_port))

        self.request_poller = zmq.Poller()
        self.request_poller.register(self.request_socket, zmq.POLLIN | zmq.POLLOUT)

        # Received via initialization
        self.is_initialized = False
        self.client_id = None
        self.server_instance_id = None
        self.environment_name = None
        self.environment_seed = None
        self.environment_id = None
        self.server_config = None
        self.command_nonce = None
        self.last_command = None

    def run(self):
        """ Run this client """
        done = False

        while not done:
            if not self.is_initialized:
                self.init()
            else:
                if self.last_command is not None:
                    done = self.run_command(self.last_command, ignore_errors=True)
                    self.last_command = None
                else:
                    command = self._fetch_command()

                    if command is not None:
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
        if self.client_id is None:
            self.logger.info(f"Worker uninitialized: waiting for environment name")
            if not self._send_initialize_request():
                return

            self.initialize_env(self.environment_name, self.environment_seed)

        self.logger.info(f"Worker {self.client_id}: waiting for registration")

        if not self._send_connect_request(self.env_space_payload()):
            return

        self.logger.info(f"Worker {self.client_id}/{self.environment_id}: properly initialized")

        self.reset_env()
        self.is_initialized = True

    def run_command(self, message, ignore_errors=False):
        """ Respond to a command received from the server """
        if message.command == pb.WorkerCommand.STEP:
            self.logger.info(f"Worker {self.client_id} received command STEP")
            self.command_nonce = message.nonce
            actions = pickle.loads(message.actions)
            self._send_frame(self.step_env(actions[self.environment_id]), ignore_errors=ignore_errors)
            return False
        elif message.command == pb.WorkerCommand.RESET:
            self.logger.info(f"Worker {self.client_id} received command RESET")
            self.command_nonce = message.nonce
            self._send_frame(self.reset_env(), ignore_errors=ignore_errors)
            return False
        elif message.command == pb.WorkerCommand.CLOSE:
            self.logger.info(f"Worker {self.client_id} received command CLOSE")
            self.close()
            return True
        elif message.command == pb.WorkerCommand.NO_COMMAND:
            return False
        else:
            raise ClientCommandException(f"Unknown command received from the server: {message}")

    def reset_client(self):
        """ Reset client state and reconnect to the server """
        self.close_env()

        self.is_initialized = False
        self.client_id = None
        self.server_instance_id = None
        self.environment_name = None
        self.environment_seed = None
        self.environment_id = None
        self.server_config = None
        self.command_nonce = None
        self.last_command = None

    def close(self):
        """ Close the client and free the resources """
        self.close_env()
        self.command_socket.close()
        self.request_socket.close()

    ####################################################################################################################
    # Requests to the server
    def _send_initialize_request(self):
        """ Request a name from the server """
        poll_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

        if self.request_socket in poll_status and poll_status[self.request_socket] == zmq.POLLOUT:
            request = pb.MasterRequest(command=pb.MasterRequest.INITIALIZE)
            self.request_socket.send(request.SerializeToString())

            response = pb.MasterResponse()
            response.ParseFromString(self.request_socket.recv())

            if response.response == pb.MasterResponse.OK:
                name_response = response.name_response

                if name_response.server_version != self.configuration.server_version:
                    raise ClientInitializationException(
                        f"Server version {name_response.server_version} does not match client version "
                        f"{self.configuration.server_version}"
                    )

                self.environment_name = name_response.name
                self.environment_seed = name_response.seed
                self.client_id = name_response.client_id
                self.server_instance_id = name_response.instance_id
                return True
            else:
                self.reset_client()
                return False
        else:
            self.reset_client()
            return False

    def _send_connect_request(self, payload):
        """ Register environment with the server """
        poll_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

        if self.request_socket in poll_status and poll_status[self.request_socket] == zmq.POLLOUT:
            request = pb.MasterRequest(
                command=pb.MasterRequest.CONNECT,
                client_id=self.client_id,
                instance_id=self.server_instance_id,
                connect_payload=payload
            )

            self.request_socket.send(request.SerializeToString())

            response = pb.MasterResponse()
            response.ParseFromString(self.request_socket.recv())

            if response.response == pb.MasterResponse.OK:
                self.environment_id = response.connect_response.environment_id

                if response.connect_response.last_command:
                    self.last_command = response.connect_response.last_command

                return True
            elif response.response == pb.MasterResponse.WAIT:
                self.logger.info(f"Worker {self.client_id} received wait command - server is busy")
                time.sleep(self.configuration.wait_period)
                return False
            else:
                self.reset_client()
                return False
        else:
            self.reset_client()
            return False

    def _send_frame(self, frame, ignore_errors=True):
        """ Send environment frame to the server """
        frame.nonce = self.command_nonce

        request = pb.MasterRequest(
            command=pb.MasterRequest.FRAME,
            client_id=self.client_id,
            instance_id=self.server_instance_id,
            frame=frame
        )

        self.request_socket.send(request.SerializeToString())
        response = pb.MasterResponse()
        response.ParseFromString(self.request_socket.recv())

        if response.response == pb.MasterResponse.OK:
            return True
        if response.response == pb.MasterResponse.SOFT_ERROR:
            return False
        else:
            if not ignore_errors:
                self.reset_client()
            return False

    def _fetch_command(self):
        """ Wait for command from the server """
        poll_status = self.command_poller.poll(self.configuration.timeout * 1000)

        if poll_status:
            message = pb.WorkerCommand()
            message.ParseFromString(self.command_socket.recv())
            return message
        else:
            self.logger.info(f"Worker {self.client_id} timeout - resetting state")
            self.reset_client()
            return None
