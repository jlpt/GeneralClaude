"""Run a list of configs x seeds, two at a time.
Usage: python sweep.py sweeps/round1.json [seeds=0,1] [parallel=2]
The sweep file is a JSON list of config-override dicts (each needs a "name")."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
configs = json.load(open(sys.argv[1]))
seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1").split(",")]
par = int(sys.argv[3]) if len(sys.argv) > 3 else 2
env = dict(os.environ, THREADS=str(max(1, 4 // par)))


def go(job):
    cfg, seed = job
    r = subprocess.run([sys.executable, os.path.join(HERE, "lab.py"), json.dumps(cfg), str(seed)],
                       capture_output=True, text=True, env=env)
    print(r.stdout.strip() or r.stderr[-2000:], flush=True)


with ThreadPoolExecutor(par) as ex:
    list(ex.map(go, [(c, s) for c in configs for s in seeds]))
