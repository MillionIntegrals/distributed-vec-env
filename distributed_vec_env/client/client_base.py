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


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences,PyTypeChecker
class ClientBase(abc.ABC):
    """
    This base class contains a stub implementation of DistributedVecEnv client.
    Ideally you should only need to override the env creation function
    """

    def __init__(self, configuration: ClientConfiguration):
        self.configuration = configuration
        self.context = zmq.Context()
        self.logger = self.configuration.logger

        self.initialization_routine()

    def initialization_routine(self):
        """ Initialize internal fields """
        self.is_initialized = False
        self.client_id = None
        self.server_instance_id = None
        self.environment_name = None
        self.environment_seed = None
        self.environment_id = None
        self.server_config = None
        self.command_nonce = None
        self.last_command = None
        self.unsuccessful_poll_count = 0

        # Idle status
        self.is_idle = False
        self.idle_timestamp = None

        # Reset status
        self.was_just_reset = False
        self.reset_frame = None
        self.reset_compensation = False

        self._create_sockets()

    def run(self):
        """ Run this client """
        done = False

        while not done:
            if not self.is_initialized:
                self.init()

                # We don't want the commands to pile up in the buffer
                if not self.is_initialized:
                    while True:
                        command = self._fetch_command_fast()

                        if command is None:
                            break
                        else:
                            self.run_command_simple(command)
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
    def reset_env(self) -> pb.Frame:
        """ Reset the environment and return next frame as Frame protocol buffer """
        pass

    @abc.abstractmethod
    def step_env(self, action) -> pb.Frame:
        """ Perform action in the env and return next frame as Frame protocol buffer. """
        pass

    @abc.abstractmethod
    def post_reset_actions(self, frame_buffer):
        """ Make any actions to the environment after the response has been sent """
        pass

    @abc.abstractmethod
    def post_step_actions(self, frame_buffer):
        """ Make any actions to the environment after the response has been sent """
        pass

    ####################################################################################################################
    # Env action wrappers
    def _perform_reset(self) -> pb.Frame:
        """ Wrap reset in a small helper utility """
        if not self.was_just_reset:
            if self.configuration.verbosity > 2:
                self.logger.info(f"Worker {self.client_id}: Performing real reset of the environment")
            self.was_just_reset = True
            self.reset_frame = self.reset_env()
            return self.reset_frame
        else:
            if self.configuration.verbosity > 2:
                self.logger.info(f"Worker {self.client_id}: Fake reset compensated")
            return self.reset_frame

    def _perform_step(self, action) -> pb.Frame:
        """ Wrap reset in a small helper utility """
        self.was_just_reset = False
        return self.step_env(action)

    ####################################################################################################################
    # Internal logic
    def init(self):
        """ Perform the initial handshake dance between the client and server to register for a mutual cooperation """
        if self.client_id is None:
            if self.configuration.verbosity > 0:
                self.logger.info(f"Worker uninitialized: waiting for environment name")

            if not self._send_initialize_request():
                return

            self.initialize_env(self.environment_name, self.environment_seed)
            # Reset env just as we get it
            self._perform_reset()

        if self.is_idle:
            command = self._fetch_command()

            if command is not None:
                self.run_command_simple(command)

            if self.is_idle and (time.time() - self.idle_timestamp) > self.configuration.timeout:
                self.is_idle = False
                self.idle_timestamp = None
        else:
            if self.configuration.verbosity > 0:
                self.logger.info(f"Worker {self.client_id}: waiting for registration")

            if not self._send_connect_request(self.env_space_payload()):
                return

            if self.configuration.verbosity > 0:
                self.logger.info(f"Worker {self.client_id}/{self.environment_id}: properly initialized")

            self._perform_reset()
            self.is_initialized = True

    def run_command(self, message) -> bool:
        """ Respond to a command received from the server """
        if message.nonce < self.command_nonce:
            if self.configuration.verbosity > 2:
                self.logger.info(f"Worker {self.client_id} ignoring command with stale nonce {message.nonce}/{self.command_nonce}")
            return False

        if message.command == pb.WorkerCommand.STEP:
            if self.configuration.verbosity > 3:
                self.logger.info(f"Worker {self.client_id} received command STEP")

            self.command_nonce = message.nonce
            actions = pickle.loads(message.actions)

            frame_buffer = self._perform_step(actions[self.environment_id])
            self._send_frame(frame_buffer)
            self.post_step_actions(frame_buffer)
            return False
        elif message.command == pb.WorkerCommand.RESET:
            self.logger.info(f"Worker {self.client_id} received command RESET")
            self.command_nonce = message.nonce

            frame_buffer = self._perform_reset()
            self._send_frame(frame_buffer)
            self.post_reset_actions(frame_buffer)
            return False
        elif message.command == pb.WorkerCommand.CLOSE:
            self.logger.info(f"Worker {self.client_id} received command CLOSE")
            self.close()
            return True
        elif message.command == pb.WorkerCommand.RESET_CLIENT:
            if self.is_initialized and self.server_instance_id != message.instance_id:
                self.logger.info(f"Worker {self.client_id} received command RESET_CLIENT")
                self.reset_client()
            return False
        elif message.command == pb.WorkerCommand.NO_COMMAND:
            return False
        elif message.command == pb.WorkerCommand.WAKE_UP:
            self.is_idle = False
            self.idle_timestamp = None
            return False
        else:
            raise ClientCommandException(f"Unknown command received from the server: {message}")

    def run_command_simple(self, command) -> bool:
        """ Run command in a possibly 'uninitialized' state just to drain the source """
        if command is not None and command.command == pb.WorkerCommand.WAKE_UP:
            self.is_idle = False
            self.idle_timestamp = None

        if command is not None and command.command == pb.WorkerCommand.RESET_CLIENT:
            if self.is_initialized and self.server_instance_id != message.instance_id:
                self.logger.info(f"Worker {self.client_id} received command RESET_CLIENT")
                self.reset_client()

    def reset_client(self):
        """ Reset client state and reconnect to the server """
        self.close()
        self.initialization_routine()

    def close(self):
        """ Close the client and free the resources """
        self.close_env()
        self.command_socket.close()
        self.request_socket.close()

    ####################################################################################################################
    # Requests to the server
    def _send_initialize_request(self):
        """ Request a name from the server """
        try:
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

                    if self.configuration.verbosity > 1:
                        self.logger.info(f"Initialization received: client id {name_response.client_id}")
                    self.environment_name = name_response.name
                    self.environment_seed = name_response.seed
                    self.client_id = name_response.client_id
                    self.server_instance_id = name_response.instance_id
                    self.reset_compensation = name_response.reset_compensation

                    return True
                else:
                    self.reset_client()
                    return False
            else:
                self.reset_client()
                return False
        except zmq.Again:
            self.reset_client()
            return False

    def _send_connect_request(self, payload):
        """ Register environment with the server """
        try:
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
                    if self.configuration.verbosity > 2:
                        self.logger.info(f"Worker {self.client_id} CONNECT - OK")
                    self.environment_id = response.connect_response.environment_id
                    self.command_nonce = 0
                    return True
                if response.response == pb.MasterResponse.OK_ENCOURAGE:
                    if self.configuration.verbosity > 2:
                        self.logger.info(f"Worker {self.client_id} CONNECT - OK ENCOURAGE")
                    self.environment_id = response.connect_response.environment_id
                    self.command_nonce = response.connect_response.last_command.nonce
                    return self._send_frame(self._perform_reset())
                elif response.response == pb.MasterResponse.WAIT:
                    if self.configuration.verbosity > 3:
                        self.logger.info(f"Worker {self.client_id} received wait command - server is busy")

                    self.is_idle = True
                    self.idle_timestamp = time.time()

                    return False
                else:
                    self.reset_client()
                    return False
            else:
                self.reset_client()
                return False
        except zmq.Again:
            self.reset_client()
            return False

    def _send_frame(self, frame):
        """ Send environment frame to the server """
        try:
            poll_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

            if self.configuration.verbosity > 3:
                self.logger.info(f"Worker {self.client_id} sending frame..")

            frame.nonce = self.command_nonce

            if self.request_socket in poll_status and poll_status[self.request_socket] == zmq.POLLOUT:
                request = pb.MasterRequest(
                    command=pb.MasterRequest.FRAME,
                    client_id=self.client_id,
                    instance_id=self.server_instance_id,
                    frame=frame
                )

                self.request_socket.send(request.SerializeToString())

                poll_recv_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

                if self.request_socket in poll_recv_status and poll_recv_status[self.request_socket] == zmq.POLLIN:
                    response = pb.MasterResponse()
                    response.ParseFromString(self.request_socket.recv())

                    if response.response == pb.MasterResponse.OK:
                        if self.configuration.verbosity > 3:
                            self.logger.info("Frame OK Response")
                        return True
                    if response.response == pb.MasterResponse.RESET:
                        if self.configuration.verbosity > 3:
                            self.logger.info("Frame RESET Response")
                        self.is_initialized = False
                        self.environment_id = None
                        self.is_idle = None
                        return False
                    if response.response == pb.MasterResponse.SOFT_ERROR:
                        if self.configuration.verbosity > 3:
                            self.logger.info("Frame SOFT_ERROR Response")
                        return True
                    else:
                        if self.configuration.verbosity > 3:
                            self.logger.info("Frame UNKNOWN Response: {}".format(response))
                        self.reset_client()
                        return False
                else:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame response timeout receive")
                    self.reset_client()
                    return False
            else:
                if self.configuration.verbosity > 3:
                    self.logger.info("Frame response timeout send")
                self.reset_client()
                return False
        except zmq.Again:
            self.reset_client()
            return False

    def _send_heartbeat_request(self):
        """ Make a request to the server, checking if it's alive """
        try:
            poll_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

            if self.configuration.verbosity > 3:
                self.logger.info(f"Worker {self.client_id} sending heartbeat..")

            if self.request_socket in poll_status and poll_status[self.request_socket] == zmq.POLLOUT:
                request = pb.MasterRequest(
                    command=pb.MasterRequest.HEARTBEAT,
                    client_id=self.client_id,
                    instance_id=self.server_instance_id,
                )

                self.request_socket.send(request.SerializeToString())

                response = pb.MasterResponse()
                response.ParseFromString(self.request_socket.recv())

                if response.response == pb.MasterResponse.OK:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame OK Response")
                    return True
                if response.response == pb.MasterResponse.RESET:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame RESET Response")
                    self.is_initialized = False
                    self.environment_id = None
                    self.is_idle = None
                    return False
                if response.response == pb.MasterResponse.SOFT_ERROR:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame SOFT_ERROR Response")
                    return True
                if response.response == pb.MasterResponse.ERROR:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame ERROR Response")
                    self.reset_client()
                    return False
                else:
                    if self.configuration.verbosity > 3:
                        self.logger.info("Frame UNKNOWN Response: {}".format(response))
                    self.reset_client()
                    return False
            else:
                self.reset_client()
                return False

        except zmq.Again:
            self.reset_client()
            return False

    def _fetch_command(self):
        """ Wait for command from the server """
        poll_status = self.command_poller.poll(self.configuration.timeout * 1000)

        if self.configuration.verbosity > 2:
            self.logger.info(f"Worker {self.client_id} polling for commands[idle={self.is_idle}]..")

        if poll_status:
            self.unsuccessful_poll_count = 0
            message = pb.WorkerCommand()
            message.ParseFromString(self.command_socket.recv())
            if self.configuration.verbosity > 3:
                self.logger.info("Received command {}".format(message.command))
            return message
        else:
            self.unsuccessful_poll_count += 1

            if self.unsuccessful_poll_count >= self.configuration.polling_limit:
                if self.configuration.verbosity > 2:
                    self.logger.info("Polling refresh")
                self._command_socket_refresh()

            return None

    def _fetch_command_fast(self):
        """ Wait for command from the server """
        poll_status = self.command_poller.poll(0)

        if self.configuration.verbosity > 2:
            self.logger.info(f"Worker {self.client_id} FAST polling for commands[idle={self.is_idle}]..")

        if poll_status:
            self.unsuccessful_poll_count = 0
            message = pb.WorkerCommand()
            message.ParseFromString(self.command_socket.recv())
            if self.configuration.verbosity > 3:
                self.logger.info("Received command {}".format(message.command))
            return message
        else:
            return None

    def _create_sockets(self):
        """ Create sockets for client-server connection """
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

        self.request_socket.setsockopt(zmq.RCVTIMEO, self.configuration.timeout * 1000)

    def _command_socket_refresh(self):
        """ Refresh command socket if polling failed enough times """
        self.command_socket.close()
        self.command_socket = self.context.socket(zmq.SUB)
        self.command_socket.setsockopt(zmq.LINGER, self.configuration.linger_period * 1000)
        self.command_socket.connect('{}:{}'.format(self.configuration.server_url, self.configuration.command_port))
        self.command_socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.command_poller = zmq.Poller()
        self.command_poller.register(self.command_socket, zmq.POLLIN)

        self._send_heartbeat_request()
