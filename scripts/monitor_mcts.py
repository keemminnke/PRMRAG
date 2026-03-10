#!/usr/bin/env python3
"""Monitor MCTS DPO build progress."""
import json, os, time, sys

CACHE = "outputs/mcts_tree_cache.jsonl"
META = CACHE + ".meta"

while True:
    # Checkpoint status
    if os.path.exists(META):
        with open(META) as f:
            meta = json.load(f)
        rollout = meta["completed_rollout"]
        nodes = meta["num_nodes"]
        mtime = os.path.getmtime(META)
        elapsed = time.time() - mtime
        ago = f"{elapsed:.0f}s ago"

        # Estimate speed
        if rollout > 0:
            # Read file creation time for rate
            ctime = os.path.getctime(CACHE)
            total_elapsed = mtime - ctime if mtime > ctime else 0
            if total_elapsed > 0:
                rate = rollout / total_elapsed * 3600  # rollouts/hr
                remaining = (64 - rollout) / (rollout / total_elapsed) if rollout < 64 else 0
                hrs = remaining / 3600
                print(f"[{time.strftime('%H:%M:%S')}] Rollout {rollout}/64 | "
                      f"{nodes:,} nodes | {rate:.1f} rollouts/hr | "
                      f"ETA: {hrs:.1f}h | updated {ago}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Rollout {rollout}/64 | "
                      f"{nodes:,} nodes | updated {ago}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Rollout {rollout}/64 | {nodes:,} nodes")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] Waiting for first checkpoint...")

    # GPU memory
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        for line in r.stdout.strip().split("\n"):
            idx, used, total = line.split(", ")
            pct = int(used)/int(total)*100
            print(f"  GPU {idx}: {used}/{total} MiB ({pct:.0f}%)")
    except:
        pass

    print()
    sys.stdout.flush()
    time.sleep(60)
