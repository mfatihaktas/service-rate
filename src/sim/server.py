import simpy

from src.sim import (
    node,
    request as request_module,
)
from src.debug_utils import *


class Server(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        sink: node.Node,
    ):
        super().__init__(env=env, _id=_id)
        self.sink = sink

        self.request_in_serv = None
        self.serv_start_time = None
        self.request_store = simpy.Store(env)
        self.recv_tasks_proc = env.process(self.recv_tasks())

    def __repr__(self):
        return f"Server(id= {self._id})"

    def repr_w_state(self):
        return (
            "Server( \n"
            f"\t num_tasks_left= {self.num_tasks_left()} \n"
            f"\t work_left= {self.work_left()} \n"
            ")"
        )

    def num_tasks_left(self) -> int:
        return len(self.request_store.items) + int(self.request_in_serv is not None)

    def work_left(self) -> float:
        remaining_serv_time = 0
        if self.request_in_serv:
            remaining_serv_time = self.request_in_serv.service_time - (
                self.env.now - self.serv_start_time
            )

        return remaining_serv_time + sum(
            request.service_time for request in self.request_store.items
        )

    def put(self, request: request_module.Request):
        slog(DEBUG, self.env, self, "recved", request=request)

        request.node_id = self._id
        self.request_store.put(request)

    def recv_tasks(self):
        slog(DEBUG, self.env, self, "started")

        num_requests_served = 0
        while True:
            self.request_in_serv = yield self.request_store.get()
            self.serv_start_time = self.env.now
            yield self.env.timeout(self.request_in_serv.service_time)

            num_requests_served += 1
            slog(
                DEBUG,
                self.env,
                self,
                "processed",
                request_in_serv=self.request_in_serv,
                num_requests_served=num_requests_served,
                queue_len=len(self.request_store.items),
            )

            self.sink.put(self.request_in_serv)
            self.request_in_serv = None

        slog(DEBUG, self.env, self, "done")


class ServerWithFiniteQueue(Server):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        sink: node.Node,
        queue_length: int,
    ):
        super().__init__(env=env, _id=_id, sink=sink)
        self.queue_length = queue_length

        self.num_dropped_requests = 0

    def __repr__(self):
        return f"ServerWithFiniteQueue(id= {self._id})"

    def put(self, request: request_module.Request):
        slog(DEBUG, self.env, self, "recved", request=request)

        request.node_id = self._id
        if len(self.request_store.items) == self.queue_length:
            slog(DEBUG, self.env, self, "queue is full, dropping", request=request)
            self.num_dropped_requests += 1
        else:
            self.request_store.put(request)
