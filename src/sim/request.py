class Request:
    def __init__(
        self,
        _id: str,
        service_time: float,
        arrival_time: float,
    ):
        self._id = _id
        self.service_time = service_time
        self.arrival_time = arrival_time

        self.node_id = None

    def __repr__(self):
        return f"Request(id= {self._id}, service_time= {self.service_time})"
