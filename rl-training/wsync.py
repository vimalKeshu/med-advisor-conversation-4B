from __future__ import annotations
import torch

# ---------------------------------------------------------------------------
# GPU 0 -- mixed into vLLM's worker via AsyncEngineArgs(worker_extension_cls=...)
# ---------------------------------------------------------------------------

class WeightSync:

    def init_weight_sync(self, host: str, port: int, rank: int, world_size: int) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        self._pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size)
        self._comm = PyNcclCommunicator(self._pg, device=self.device)

    def recv_weights(self, metas: list[tuple[str, list[int]]]) -> None:
        stream = torch.cuda.current_stream()
        for name, shape in metas:
            buffer = torch.empty(tuple(shape), dtype=torch.bfloat16, device=self.device)
            self._comm.broadcast(buffer, src=0, stream=stream)
            self.model_runner.model.load_weights([(name, buffer)])
        self._pg.barrier()      # one clean handoff point, not 398


# ---------------------------------------------------------------------------
# GPU 1 -- trainer side
# ---------------------------------------------------------------------------

def weight_metas(model) -> list[tuple[str, list[int]]]:
    return [(name, list(p.shape)) for name, p in model.named_parameters()]

class WeightSender:

    def __init__(self, model, host: str, port: int,
                 rank: int = 0, world_size: int = 2) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        self.model = model
        self.pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size)
        self.comm = PyNcclCommunicator(
            self.pg, device=next(model.parameters()).device)

    def metas(self) -> list[tuple[str, list[int]]]:
        return weight_metas(self.model)

    def send(self) -> None:
        stream = torch.cuda.current_stream()
        for _, parameter in self.model.named_parameters():
            self.comm.broadcast(parameter.data.to(torch.bfloat16), src=0, stream=stream)
        self.pg.barrier()


# ---------------------------------------------------------------------------
# smoke test -- validates the rendezvous and the NVLink path, no model involved
# ---------------------------------------------------------------------------

def _receiver(device: str, host: str, port: int, n: int, ok) -> None:
    """Module level, not a closure: spawn pickles the target by qualname."""
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=host, port=port, rank=1, world_size=2)
    comm = PyNcclCommunicator(pg, device=torch.device(device))
    good = True
    for i in range(n):
        buffer = torch.empty(1024, dtype=torch.bfloat16, device=device)
        comm.broadcast(buffer, src=0)
        pg.barrier()
        good &= bool((buffer.float() == float(i)).all())
    ok.value = int(good)

def _smoke(host: str = "127.0.0.1", port: int = 51216, n: int = 4) -> int:
    """rank 1 on cuda:0, rank 0 on cuda:1. No CUDA_VISIBLE_DEVICES here -- torch
    is already imported by the time either side could set it. The real actor and
    trainer get device isolation from run.py, which sets the env before any
    child imports torch."""
    import multiprocessing as mp

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if torch.cuda.device_count() < 2:
        print(f"need 2 GPUs, found {torch.cuda.device_count()}")
        return 1

    ctx = mp.get_context("spawn")
    ok = ctx.Value("i", 0)
    proc = ctx.Process(target=_receiver, args=("cuda:0", host, port, n, ok))
    proc.start()

    try:
        pg = StatelessProcessGroup.create(host=host, port=port, rank=0, world_size=2)
        comm = PyNcclCommunicator(pg, device=torch.device("cuda:1"))
        for i in range(n):
            comm.broadcast(
                torch.full((1024,), float(i), dtype=torch.bfloat16, device="cuda:1"),
                src=0,
            )
            pg.barrier()
    finally:
        proc.join(timeout=180)
        if proc.is_alive():
            proc.terminate()
            print("NCCL smoke: HANG (receiver did not finish in 180s)")
            return 1

    if proc.exitcode != 0:
        print(f"NCCL smoke: FAIL (receiver exited {proc.exitcode})")
        return 1
    print("NCCL smoke:", "PASS" if ok.value else "FAIL (contents mismatched)")
    return 0 if ok.value else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="ignored; the only mode")
    parser.add_argument("--port", type=int, default=51216)
    args = parser.parse_args()
    raise SystemExit(_smoke(port=args.port))