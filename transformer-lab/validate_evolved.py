"""Re-test evolved genomes on fresh seeds, at proxy size and at full lab size.
Usage: python validate_evolved.py proxy|full"""
import json, os, sys
import multiprocessing as mp
import evolve

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = dict(d=128, n_layer=4, n_head=4, ctx=128, batch=32, steps=2000, lr=3e-3, eval_batches=100)


def load(mode, i):
    for line in open(os.path.join(HERE, f"evolution_{mode}.jsonl")):
        h = json.loads(line)
        if h["i"] == i:
            return h["genome"]


CANDIDATES = {
    "transformer (hand-designed)": evolve.TRANSFORMER,
    "evolved from transformer #1 (eval 185)": load("seeded", 185),
    "evolved from transformer #2 (eval 20)": load("seeded", 20),
    "evolved from scratch #1 (eval 331)": load("scratch", 331),
}


def job(args):
    name, seed, size = args
    P = evolve.PROXY if size == "proxy" else FULL
    v, n, secs = evolve.fitness(CANDIDATES[name], P=P, seed=seed)
    return {"name": name, "size": size, "seed": seed, "val_loss": round(v, 4), "params": n, "secs": round(secs)}


if __name__ == "__main__":
    size = sys.argv[1]
    seeds = [1, 2, 3] if size == "proxy" else [0, 1]
    os.environ["THREADS"] = "1" if size == "proxy" else "2"
    jobs = [(n, s, size) for n in CANDIDATES for s in seeds]
    with mp.Pool(4 if size == "proxy" else 2) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(json.dumps(r), flush=True)
            with open(os.path.join(HERE, "evolved_validation.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
