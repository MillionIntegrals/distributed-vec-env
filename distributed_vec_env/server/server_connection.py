import logging
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


class ServerConnection:
    """ An object encapsulating server network interface """

    def __init__(self, configuration: ServerConfiguration):
        self.context = zmq.Context()
        self.configuration = configuration
        self.logger = logging.getLogger(__name__)

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

        self.logger.info("Master: awaiting initialization")

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
            nonce=numpy_util.random_int64()
        )

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

        command = pb.WorkerCommand(
            command=pb.WorkerCommand.STEP,
            nonce=numpy_util.random_int64(),
            actions=actions
        )

        self._publish_command(command)
        self._reset_frame_buffer()

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
                unregistered_any = False

                for idx, frame in enumerate(self.observation_buffer):
                    if frame is None and idx in self.env_client_map:
                        self._unregister_env(idx)
                        unregistered_any = True

                        prev_observation = self.prev_observation_buffer[idx]

                        if prev_observation is None:
                            # No previous observations, just start everything from scratch
                            self._reset_communication_state()
                            self._resend_last_command()
                        else:
                            self.observation_buffer[idx] = self.prev_observation_buffer[idx]

                            self.done_buffer[idx] = True
                            self.reward_buffer[idx] = 0.0

                start_time = time.time()

                if not unregistered_any:
                    self._reset_communication_state()
                    self._resend_last_command()

        self.last_command = None

        # For now we don't care about infos. That may change
        infos = [{} for _ in range(self.number_of_clients)]
        result = np.stack(self.observation_buffer, axis=0), np.stack(self.reward_buffer), np.stack(self.done_buffer), infos

        self._reset_frame_buffer()

        return result

    def close_environments(self):
        """ Close all child environments """
        if self.is_closed:
            raise ServerClosedException("Environment already closed")

        self.is_closed = True

        command = pb.WorkerCommand(
            command=pb.WorkerCommand.CLOSE,
            nonce=numpy_util.random_int64()
        )

        self._publish_command(command)

        self.request_socket.close()
        self.command_socket.close()

    ####################################################################################################################
    # Communication bits
    def _reset_communication_state(self):
        """ Reset client-server state """
        self.client_env_map = {}
        self.env_client_map = {}

        self.prev_observation_buffer = [None for _ in range(self.number_of_clients)]
        self.observation_buffer = [None for _ in range(self.number_of_clients)]
        self.reward_buffer = [None for _ in range(self.number_of_clients)]
        self.done_buffer = [None for _ in range(self.number_of_clients)]

        self.command_nonce = None

    def _resend_last_command(self):
        """ Send again last command """
        self.last_command.nonce = numpy_util.random_int64()
        self._publish_command(self.last_command)

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
                instance_id=self.instance_id
            )
        )

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
                    response=pb.MasterResponse.OK,
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

            self.logger.info(f"Master: assigned client id {request.client_id} to environment {new_environment_id}")
            self.logger.info(f"Client env map: {self.client_env_map}")

            self.client_env_map[request.client_id] = new_environment_id
            self.env_client_map[new_environment_id] = request.client_id

            self.request_socket.send(response.SerializeToString())
        else:
            # Too many clients connected, ignore for now
            response = pb.MasterResponse(response=pb.MasterResponse.WAIT)
            self.request_socket.send(response.SerializeToString())

    def _map_new_env(self, client_id):
        """ Map client to a free environment slot """
        for i in range(self.number_of_clients):
            if i not in self.env_client_map:
                self.env_client_map[i] = client_id
                self.client_env_map[client_id] = i
                return i

        return None

    def _unregister_env(self, env_id):
        """ Map client to a free environment slot """
        if env_id in self.env_client_map:
            self.logger.info(f"Unregistering stale environment {env_id}")
            client_id = self.env_client_map[env_id]

            del self.env_client_map[env_id]
            del self.client_env_map[client_id]

    def _handle_frame_request(self, request):
        """ Handle the FRAME request from the client """
        frame = request.frame

        if request.client_id not in self.client_env_map or self.env_client_map[self.client_env_map[request.client_id]] != request.client_id:
            self.logger.info(f"Received frame with stale client {request.client_id}")
            self.logger.info(self.client_env_map)
            self.logger.info(self.env_client_map)

            response = pb.MasterResponse(response=pb.MasterResponse.ERROR)
            self.request_socket.send(response.SerializeToString())
        elif frame.nonce != self.command_nonce:
            self.logger.info(f"Received frame with incorrect nonce")
            # Notify client of an error request
            response = pb.MasterResponse(response=pb.MasterResponse.SOFT_ERROR)
            self.request_socket.send(response.SerializeToString())

        else:
            environment_id = self.client_env_map[request.client_id]

            self.logger.info(f"Received frame with correct nonce from {request.client_id}/{environment_id}")

            self.observation_buffer[environment_id] = numpy_util.deserialize_numpy(frame.observation)
            self.reward_buffer[environment_id] = frame.reward
            self.done_buffer[environment_id] = frame.done

            # Just confirm receipt, nothing more
            response = pb.MasterResponse(
                response=pb.MasterResponse.OK
            )
            self.request_socket.send(response.SerializeToString())

    def _publish_command(self, command):
        """ Send a command to all the clients """
        self.command_nonce = command.nonce
        self.last_command = command
        self.command_socket.send(command.SerializeToString())
