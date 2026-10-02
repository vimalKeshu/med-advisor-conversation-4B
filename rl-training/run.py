#!/usr/bin/env python3
"""Launcher: actor on GPU 0, trainer on GPU 1, two queues between them.

STDLIB ONLY at module scope. Under spawn the child re-imports this module before
unpickling its target, so a top-level `import torch` here would initialise torch
in both children before either could set CUDA_VISIBLE_DEVICES -- and both would
land on GPU 0.

    python run.py                       # full async run
    python run.py --config other.yaml
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _child(role: str, config_path: str, traj_q, ctrl_q) -> int:
    """Module level so spawn can pickle it by qualname. Sets the device before
    anything imports torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if role == "actor" else "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from common import load_config

    full = load_config(config_path)
    if role == "actor":
        from rollout import actor_main
        return actor_main(full, traj_q, ctrl_q)
    from trainer import trainer_main
    return trainer_main(full, traj_q, ctrl_q)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    # Unbounded is safe: the actor produces exactly groups_per_step per batch and
    # then blocks on the sync, so depth is bounded by the protocol, not a maxsize.
    traj_q = ctx.Queue()
    ctrl_q = ctx.Queue()

    procs = {
        role: ctx.Process(target=_child,
                          args=(role, args.config, traj_q, ctrl_q),
                          name=role)
        for role in ("actor", "trainer")
    }
    print("actor -> GPU 0, trainer -> GPU 1", flush=True)
    for proc in procs.values():
        proc.start()

    try:
        while True:
            dead = [name for name, p in procs.items() if not p.is_alive()]
            if dead:
                print(f"{dead[0]} exited ({procs[dead[0]].exitcode}); stopping the other",
                      flush=True)
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\ninterrupted; stopping both", flush=True)
        ctrl_q.put(("stop",))            # lets the actor exit its own loop first
        time.sleep(5.0)

    for name, proc in procs.items():
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
        print(f"{name}: exit {proc.exitcode}", flush=True)

    # A Queue with buffered items keeps its feeder thread alive and hangs exit.
    for q in (traj_q, ctrl_q):
        q.cancel_join_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())