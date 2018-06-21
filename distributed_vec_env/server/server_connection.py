import numpy as np
import pickle
import zmq
import time

import distributed_vec_env.messages.protocol_pb2 as pb
import distributed_vec_env.numpy_util as numpy_util


from .server_configuration import ServerConfiguration


class ServerHandlerException(Exception):
    """ Exception during server request handling """
    pass


class ServerClosedException(Exception):
    """ Exception during server request handling """
    pass


# noinspection PyUnresolvedReferences,PyTypeChecker
class ServerConnection:
    """ An object encapsulating server network interface """

    def __init__(self, configuration: ServerConfiguration):
        self.context = zmq.Context()
        self.configuration = configuration
        self.logger = self.configuration.logger

        # Pull socket - workers send work results over this channel
        self.request_socket = self.context.socket(zmq.REP)
        self.request_socket.setsockopt(zmq.LINGER, self.configuration.linger_period * 1000)
        self.request_socket.bind("{}:{}".format(self.configuration.server_url, self.configuration.request_port))

        self.request_poller = zmq.Poller()
        self.request_poller.register(self.request_socket, zmq.POLLIN)

        # Push socket - send commands to workers over this channel
        self.command_socket = self.context.socket(zmq.PUB)
        self.command_socket.setsockopt(zmq.LINGER, self.configuration.linger_period * 1000)
        self.command_socket.bind("{}:{}".format(self.configuration.server_url, self.configuration.command_port))

        # Some internal state
        self.last_client_id_assigned = 0

        self.observation_space = None
        self.action_space = None

        self.is_closed = False
        self.instance_id = numpy_util.random_int64()

        self.client_env_map = {}
        self.env_client_map = {}

        self.prev_observation_buffer = [None for _ in range(self.number_of_clients)]
        self.observation_buffer = [None for _ in range(self.number_of_clients)]
        self.reward_buffer = [None for _ in range(self.number_of_clients)]
        self.done_buffer = [None for _ in range(self.number_of_clients)]
        self.info_buffer = [None for _ in range(self.number_of_clients)]

        self.last_command_nonce = np.int64(0)
        self.command_nonce = None
        self.last_command = None

    ####################################################################################################################
    # Data access
    @property
    def number_of_clients(self):
        """ Number of environments/clients this server supports """
        return self.configuration.number_of_environments

    @property
    def connected_clients(self):
        """ Return number of connected clients """
        return len(self.client_env_map)

    ####################################################################################################################
    # Frame buffer management
    def _reset_frame_buffer(self):
        """ Reset the frame buffer to initial state """
        self.prev_observation_buffer = self.observation_buffer
        self.observation_buffer = [None for _ in range(self.number_of_clients)]
        self.reward_buffer = [None for _ in range(self.number_of_clients)]
        self.done_buffer = [None for _ in range(self.number_of_clients)]

    def _is_frame_buffer_ready(self):
        """ Is frame buffer full and ready to send back """
        return all(x is not None for x in self.observation_buffer)

    ####################################################################################################################
    # External interface
    def initialize(self):
        """ Perform initial handshake with all the clients and make sure there are enough to proceed """
        if self.connected_clients > 0:
            raise ServerHandlerException("Server already initialized")

        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        if self.configuration.verbosity > 0:
            self.logger.info("Master: awaiting initialization")

        self.send_client_reset()

        while self.connected_clients < self.number_of_clients:
            self._communication_loop()

        return self.observation_space, self.action_space

    def reset_environments(self):
        """
        Reset all underlying client environments.
        Returns an array of observations
        """
        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        command = pb.WorkerCommand(
            command=pb.WorkerCommand.RESET,
            # nonce=numpy_util.random_int64()
            nonce=self.last_command_nonce
        )

        self.last_command_nonce += 1

        self._publish_command(command)
        self._reset_frame_buffer()

        obs, _, _, _ = self.gather_frames()
        return obs

    def send_actions(self, actions):
        """
        Send actions to the environments.
        Wait for their responses.
        """
        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        action_bytes = pickle.dumps(actions)

        command = pb.WorkerCommand(
            command=pb.WorkerCommand.STEP,
            # nonce=numpy_util.random_int64(),
            nonce=self.last_command_nonce,
            actions=action_bytes
        )
        self.last_command_nonce += 1

        self._publish_command(command)
        self._reset_frame_buffer()

    def send_client_reset(self):
        """ Send worker command to reset the environments if they have been already initialized """
        command = pb.WorkerCommand(
            command=pb.WorkerCommand.RESET_CLIENT,
            # nonce=numpy_util.random_int64(),
            nonce=self.last_command_nonce,
            instance_id=self.instance_id
        )
        self.last_command_nonce += 1
        self._publish_command(command)

    def gather_frames(self):
        """
        Wait until all the clients come back with valid frames.

        Returns (obs, rews, dones, infos):
         - obs: an array of observations
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: an array of info objects
        """
        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        start_time = time.time()

        while not self._is_frame_buffer_ready():
            self._communication_loop()

            # If we're timing out, unregister old environments
            if time.time() - start_time > self.configuration.timeout:
                self._communication_timeout()
                start_time = time.time()

        self.last_command = None

        result = (
            np.stack(self.observation_buffer, axis=0), np.stack(self.reward_buffer), np.stack(self.done_buffer),
            self.info_buffer
        )

        self._reset_frame_buffer()

        return result

    def close_environments(self):
        """ Close all child environments """
        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        self.is_closed = True

        command = pb.WorkerCommand(
            command=pb.WorkerCommand.CLOSE,
            # nonce=numpy_util.random_int64()
            nonce=self.last_command_nonce

        )
        self.last_command_nonce += 1

        self._publish_command(command)

        self.request_socket.close()
        self.command_socket.close()

    ####################################################################################################################
    # Communication bits
    def _communication_timeout(self):
        """ Decision loop triggered when some of the workers didn't submit the data on time """
        for idx, frame in enumerate(self.observation_buffer):
            if frame is None and idx in self.env_client_map:
                self._unregister_env(idx)

    def _communication_loop(self):
        """
        Main function of the server. Actively listen for connections from the clients and
        respond to their requests
        """
        poll_status = dict(self.request_poller.poll(self.configuration.timeout * 1000))

        if poll_status:
            request = pb.MasterRequest()
            request.ParseFromString(self.request_socket.recv())

            if request.command != pb.MasterRequest.INITIALIZE and request.instance_id != self.instance_id:
                # Received request for the wrong server
                response = pb.MasterResponse(response=pb.MasterResponse.ERROR)
                self.request_socket.send(response.SerializeToString())
            else:
                if request.command == pb.MasterRequest.INITIALIZE:
                    self._handle_initialize_request()
                elif request.command == pb.MasterRequest.CONNECT:
                    self._handle_connect_request(request)
                elif request.command == pb.MasterRequest.FRAME:
                    self._handle_frame_request(request)
                elif request.command == pb.MasterRequest.HEARTBEAT:
                    self._handle_heartbeat_request(request)
                else:
                    raise ServerHandlerException(f"Received unknown request: {request}")

    def _handle_initialize_request(self):
        """ Handle the INITIALIZE request from the client """
        response = pb.MasterResponse(
            response=pb.MasterResponse.OK,
            name_response=pb.NameResponse(
                name=self.configuration.environment_name,
                seed=self.last_client_id_assigned,
                server_version=self.configuration.server_version,
                client_id=self.last_client_id_assigned,
                instance_id=self.instance_id,
                reset_compensation=self.configuration.reset_compensation
            )
        )

        if self.configuration.verbosity > 0:
            self.logger.info(f"Master: assigned client id {self.last_client_id_assigned}")

        # Client ids are monotonically increasing
        self.last_client_id_assigned += 1

        self.request_socket.send(response.SerializeToString())

    def _handle_connect_request(self, request):
        """ Handle the CONNECT request from the client """
        if self.observation_space is None or self.action_space is None:
            self.observation_space, self.action_space = pickle.loads(request.connect_payload.spaces)

        if self.connected_clients < self.number_of_clients:
            # Accept new client
            new_environment_id = self._map_new_env(request.client_id)

            if self.last_command is not None:
                response = pb.MasterResponse(
                    # Encourage the client to send a reset frame straight away
                    response=pb.MasterResponse.OK_ENCOURAGE,
                    connect_response=pb.ConnectResponse(
                        environment_id=new_environment_id,
                        last_command=self.last_command
                    )
                )
            else:
                response = pb.MasterResponse(
                    response=pb.MasterResponse.OK,
                    connect_response=pb.ConnectResponse(
                        environment_id=new_environment_id
                    )
                )

            if self.configuration.verbosity > 1:
                self.logger.info(f"Master: assigned client id c{request.client_id} to environment e{new_environment_id}")

            if self.configuration.verbosity > 2:
                self.logger.info(f"Client env map: {self.client_env_map}")

            self.client_env_map[request.client_id] = new_environment_id
            self.env_client_map[new_environment_id] = request.client_id

            self.request_socket.send(response.SerializeToString())
        else:
            # Too many clients connected, ignore for now
            response = pb.MasterResponse(response=pb.MasterResponse.WAIT)
            self.request_socket.send(response.SerializeToString())

    def _handle_heartbeat_request(self, _):
        """ Respond to the client that the server is alive """
        response = pb.MasterResponse(response=pb.MasterResponse.OK)
        self.request_socket.send(response.SerializeToString())

    def _map_new_env(self, client_id):
        """ Map client to a free environment slot """
        for i in range(self.number_of_clients):
            if i not in self.env_client_map:
                self.env_client_map[i] = client_id
                self.client_env_map[client_id] = i
                return i

        raise RuntimeError("Cannot map environment, all are busy!")

    def _unregister_env(self, env_id):
        """ Map client to a free environment slot """
        if env_id in self.env_client_map:
            client_id = self.env_client_map[env_id]
            if self.configuration.verbosity > 1:
                self.logger.info(f"Unregistering environment c{client_id}/e{env_id}")

            del self.env_client_map[env_id]
            del self.client_env_map[client_id]

    def _handle_frame_request(self, request):
        """ Handle the FRAME request from the client """
        frame = request.frame

        if (request.client_id not in self.client_env_map) or \
                (self.env_client_map[self.client_env_map[request.client_id]] != request.client_id):
            if self.configuration.verbosity > 3:
                self.logger.info(f"Received frame with stale client c{request.client_id}")

            response = pb.MasterResponse(response=pb.MasterResponse.ERROR)
            self.request_socket.send(response.SerializeToString())
        elif frame.nonce != self.command_nonce:
            if self.configuration.verbosity > 2:
                self.logger.info(
                    f"Received frame with incorrect nonce from client c{request.client_id} ({frame.nonce, self.command_nonce})"
                )
            # Notify client of an error request
            response = pb.MasterResponse(response=pb.MasterResponse.SOFT_ERROR)
            self.request_socket.send(response.SerializeToString())
        else:
            # CORRECT FRAME RECEIVED
            environment_id = self.client_env_map[request.client_id]

            if self.configuration.verbosity > 3:
                self.logger.info(f"Received frame with correct nonce from c{request.client_id}/e{environment_id}")

            self.observation_buffer[environment_id] = numpy_util.deserialize_numpy(frame.observation)
            self.reward_buffer[environment_id] = frame.reward
            self.done_buffer[environment_id] = frame.done
            self.info_buffer[environment_id] = pickle.loads(frame.info)

            if frame.done and self.configuration.reset_compensation:
                # Reset compensation, unregister the resetting environment
                if self.configuration.verbosity > 2:
                    self.logger.info(f"Reset compensation: unregistering env c{request.client_id}/e{environment_id}")

                self._unregister_env(environment_id)

                response = pb.MasterResponse(
                    response=pb.MasterResponse.RESET
                )
                self.request_socket.send(response.SerializeToString())

                self._send_wake_up_call()
            else:
                # Just confirm receipt, nothing more
                response = pb.MasterResponse(
                    response=pb.MasterResponse.OK
                )
                self.request_socket.send(response.SerializeToString())

    def _send_wake_up_call(self):
        """ Send a WAKE_UP message to all the clients """
        command = pb.WorkerCommand(
            command=pb.WorkerCommand.WAKE_UP,
            # nonce=numpy_util.random_int64(),
            # nonce=self.last_command_nonce
        )
        # self.last_command_nonce += 1
        self._publish_command(command, remember=False)

    def _publish_command(self, command, remember=True):
        """ Send a command to all the clients """
        if remember:
            self.command_nonce = command.nonce
            self.last_command = command

        self.command_socket.send(command.SerializeToString())
