import zmq
import pickle
import logging

import distributed_vec_env.messages.protocol_pb2 as pb
import distributed_vec_env.numpy_util as numpy_util

from .server_configuration import ServerConfiguration


class ServerHandlerException(Exception):
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
        self.request_socket.bind("{}:{}".format(self.configuration.server_url, self.configuration.request_port))

        # Push socket - send commands to workers over this channel
        self.command_socket = self.context.socket(zmq.PUB)
        self.command_socket.bind("{}:{}".format(self.configuration.server_url, self.configuration.command_port))

        # Some internal state
        self.last_client_id_assigned = 0
        self.connected_clients = 0
        self.client_env_map = {}
        self.env_client_map = {}

        self.observation_buffer = [None for _ in range(self.number_of_clients)]
        self.reward_buffer = [None for _ in range(self.number_of_clients)]
        self.done_buffer = [None for _ in range(self.number_of_clients)]

        self.observation_space = None
        self.action_space = None
        self.is_closed = False

    ####################################################################################################################
    # Data access
    @property
    def number_of_clients(self):
        return self.configuration.number_of_environments

    ####################################################################################################################
    # Frame buffer management
    def _reset_frame_buffer(self):
        """ Reset the frame buffer to initial state """
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
        self.logger.info("Master: awaiting initialization")

        while self.connected_clients < self.number_of_clients:
            self._communication_loop()

        return self.observation_space, self.action_space

    def reset_environments(self):
        """
        Reset all underlying client environments.
        Returns an array of observations
        """
        command = pb.WorkerMessage(
            command=pb.WorkerMessage.RESET
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
        command = pb.WorkerMessage(
            command=pb.WorkerMessage.STEP,
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
        while not self._is_frame_buffer_ready():
            self._communication_loop()

        # For now we don't care about infos. That may change
        infos = [{} for _ in range(self.number_of_clients)]
        return self.observation_buffer, self.reward_buffer, self.done_buffer, infos

    def close_environments(self):
        """ Close all child environments """
        self.is_closed = True

        command = pb.WorkerMessage(
            command=pb.WorkerMessage.CLOSE,
        )

        self._publish_command(command)

    ####################################################################################################################
    # Communication bits
    def _communication_loop(self):
        """
        Main function of the server. Actively listen for connections from the clients and
        respond to their requests
        """
        request = pb.MasterRequest()
        request.ParseFromString(self.request_socket.recv())

        if request.command == pb.MasterRequest.INITIALIZATION:
            if self.observation_space is None or self.action_space is None:
                self.observation_space, self.action_space = pickle.loads(request.initialization.spaces)

            if self.connected_clients < self.number_of_clients:
                # Accept new client
                # Heuristic, for now
                new_environment_id = request.client_id

                response = pb.InitializationResponse(
                    environment_id=new_environment_id
                )

                self.logger.info(f"Master: assigned client id {request.client_id} to environment {new_environment_id}")

                self.client_env_map[request.client_id] = new_environment_id
                self.env_client_map[new_environment_id] = request.client_id
                self.connected_clients += 1

                self.request_socket.send(response.SerializeToString())
            else:
                # Too many clients connected, ignore for now
                response = pb.InitializationResponse()
                self.request_socket.send(response.SerializeToString())
        elif request.command == pb.MasterRequest.FRAME:
            frame = request.frame
            environment_id = self.client_env_map[request.client_id]

            self.observation_buffer[environment_id] = numpy_util.deserialize_numpy(frame.observation)
            self.reward_buffer[environment_id] = frame.reward
            self.done_buffer[environment_id] = frame.done

            # Just confirm receipt, nothing more
            response = pb.ConfirmationResponse()
            self.request_socket.send(response.SerializeToString())
        elif request.command == pb.MasterRequest.NAME:
            response = pb.NameResponse(
                name=self.configuration.environment_name,
                seed=self.last_client_id_assigned,
                server_version=self.configuration.server_version,
                client_id=self.last_client_id_assigned
            )

            self.logger.info(f"Master: assigned client id {self.last_client_id_assigned}")

            # Client ids are monotonically increasing
            self.last_client_id_assigned += 1

            self.request_socket.send(response.SerializeToString())
        else:
            raise ServerHandlerException(f"Received unknown request: {request}")

    def _publish_command(self, command):
        """ Send a command to all the clients """
        self.command_socket.send(command.SerializeToString())

