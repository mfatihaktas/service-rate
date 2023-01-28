import simpy

from src.sim import (
    node,
    random_variable,
    request as request_module,
)
from src.debug_utils import *


class Source(node.Node):
    def __init__(
        self,
        env: simpy.Environment,
        _id: str,
        inter_gen_time_rv: random_variable.RandomVariable,
        service_time_rv: random_variable.RandomVariable,
        next_hop: node.Node,
        num_requests_to_send: int = None,
    ):
        super().__init__(env=env, _id=_id)
        self.inter_gen_time_rv = inter_gen_time_rv
        self.service_time_rv = service_time_rv
        self.next_hop = next_hop
        self.num_requests_to_send = num_requests_to_send

        self.send_messages_proc = env.process(self.send_requests())

    def __repr__(self):
        return f"Source(id= {self._id})"

    def send_requests(self):
        slog(DEBUG, self.env, self, "started")

        request_id = 0
        while True:
            inter_msg_gen_time = self.inter_gen_time_rv.sample()
            slog(
                DEBUG, self.env, self, "waiting", inter_msg_gen_time=inter_msg_gen_time
            )
            yield self.env.timeout(inter_msg_gen_time)

            request = request_module.Request(
                _id=request_id,
                service_time=self.service_time_rv.sample(),
                arrival_time=self.env.now,
            )

            slog(DEBUG, self.env, self, "sending", request=request)
            self.next_hop.put(request)

            request_id += 1
            if self.num_requests_to_send and request_id >= self.num_requests_to_send:
                break

        slog(DEBUG, self.env, self, "started")
