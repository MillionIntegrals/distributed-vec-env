import logging


class ServerConfiguration:
    """ Configuration of network server distributing work among workers """

    def __init__(self, server_url, command_port, request_port, number_of_environments, environment_name,
                 server_version=1, timeout=30, linger_period=1, logger=None, reset_compensation=False):
        self.server_url = server_url
        self.command_port = command_port
        self.request_port = request_port
        self.number_of_environments = number_of_environments
        self.linger_period = linger_period

        self.environment_name = environment_name
        self.server_version = server_version
        self.timeout = timeout

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.reset_compensation = reset_compensation

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"  {self.server_url}, {self.command_port}, {self.request_port}, {self.number_of_environments},"
            f"  {self.environment_name}, {self.server_version}, {self.timeout}"
            ")"
        )


