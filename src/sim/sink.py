import simpy

from src.sys import request as request_module
from src.debug_utils import *


class Sink:
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        num_requests_to_recv: int = None,
    ):
        self.env = env
        self._id = _id
        self.num_requests_to_recv = num_requests_to_recv

        self.request_store = simpy.Store(env)
        self.recv_tasks_proc = env.process(self.recv_tasks())

        self.request_response_time_list = []

    def __repr__(self):
        return f"Sink(id= {self._id})"

    def put(self, request: request_module.Task):
        slog(DEBUG, self.env, self, "recved", request=request)

        self.request_store.put(request)

    def recv_tasks(self):
        slog(DEBUG, self.env, self, "started")

        num_requests_recved = 0
        while True:
            request = yield self.request_store.get()
            num_requests_recved += 1
            slog(DEBUG, self.env, self, "recved", request=request, num_requests_recved=num_requests_recved)

            if self.sching_agent:
                response_time = self.env.now - request.arrival_time
                self.request_response_time_list.append(response_time)

            if num_requests_recved >= self.num_requests_to_recv:
                slog(DEBUG, self.env, self, "recved requested # tasks", num_requests_recved=num_requests_recved)
                break

        slog(DEBUG, self.env, self, "done")
