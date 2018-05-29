class ServerConfiguration:
    """ Configuration of network server distributing work among workers """

    def __init__(self, server_url, command_port, request_port, number_of_environments, environment_name, server_version=1, timeout=30):
        self.server_url = server_url
        self.command_port = command_port
        self.request_port = request_port
        self.number_of_environments = number_of_environments

        self.environment_name = environment_name
        self.server_version = server_version
        self.timeout = timeout

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"  {self.server_url}, {self.command_port}, {self.request_port}, {self.number_of_environments},"
            f"  {self.environment_name}, {self.server_version}, {self.timeout}"
            ")"
        )


