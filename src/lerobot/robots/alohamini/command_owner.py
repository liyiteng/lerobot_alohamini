"""Single-writer command ownership, released only after the Host watchdog stops motion."""

from collections import OrderedDict


class CommandOwner:
    def __init__(self) -> None:
        self.owner: str | None = None
        self.epoch = 0
        self._sequences: OrderedDict[str, int] = OrderedDict()

    def accept(self, metadata: dict, host_session_id: str) -> bool:
        client = metadata.get("client_id", "legacy")
        if metadata and (
            type(metadata.get("control_epoch")) is not int
            or metadata["control_epoch"] != self.epoch
            or metadata.get("host_session_id") != host_session_id
        ):
            return False
        if metadata.get("host_session_id", host_session_id) != host_session_id:
            return False
        if self.owner is not None and self.owner != client:
            return False
        if metadata:
            sequence = metadata["sequence"]
            if sequence <= self._sequences.get(client, -1):
                return False
            self._sequences[client] = sequence
            self._sequences.move_to_end(client)
            if len(self._sequences) > 256:
                self._sequences.popitem(last=False)
        self.owner = client
        return True

    def release(self) -> None:
        self.owner = None
        self.epoch += 1
