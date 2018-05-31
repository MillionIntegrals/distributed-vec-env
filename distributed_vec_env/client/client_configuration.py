
class ClientConfiguration:
    """ Configuration of network worker evaluating the environments """

    def __init__(self, server_url, command_port, request_port, server_version=1, timeout=30, wait_period=10,
                 linger_period=1):
        self.server_url = server_url
        self.command_port = command_port
        self.request_port = request_port
        self.server_version = server_version
        self.timeout = timeout
        self.wait_period = wait_period
        self.linger_period = linger_period
