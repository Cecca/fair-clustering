import h5py
import hashlib
import os
import logging
import sqlite3
import assess
import datasets
import json


def _hash_dict_update(state, d):
    def _up(x):
        if not isinstance(x, str):
            x = str(x)
        x = x.encode('utf8')
        state.update(x)

    pairs = sorted(list(d.items()))
    for k, v in pairs:
        _up(k)
        _up(v)


def run_hash(dataset, algorithm, k, delta, attrs_dict):
    state = hashlib.sha256()
    state.update(dataset.encode("utf8"))
    state.update(algorithm.encode("utf8"))
    state.update(bytes(str(k), encoding="utf8"))
    state.update(bytes(str(delta), encoding="utf8"))
    _hash_dict_update(state, attrs_dict)
    return state.hexdigest()


def compute_key(dataset, algorithm, k, delta, attrs_dict):
    hash = run_hash(dataset, algorithm, k, delta, attrs_dict)
    key = f"{dataset}/{algorithm}/{k}/{delta}/{hash}"
    return key


def write_clustering(opath, centers, assignment, dataset, algorithm, k, delta, attrs_dict):
    mode = "r+" if os.path.isfile(opath) else "w"
    with h5py.File(opath, mode) as hfp:
        key = compute_key(dataset, algorithm, k, delta, attrs_dict)
        if key in hfp:
            logging.warn(
                "Name already exists in result file, skipping: %s", key)
            return
        group = hfp.create_group(key)
        group["centers"] = centers
        group["assignment"] = assignment
        for k, v in attrs_dict.items():
            group.attrs[k] = v
        return key


def read_clustering(opath, dataset, algorithm, k, attrs_dict):
    with h5py.File(opath, "r") as hfp:
        key = compute_key(dataset, algorithm, k, attrs_dict)
        if key not in hfp:
            logging.error("Name missing in result file: %s", key)
            raise Exception("Name missing in result file")
        centers = hfp[f"{key}/centers"][:]
        assignment = hfp[f"{key}/assignment"][:]
        return centers, assignment


def get_db():
    db = sqlite3.connect("results.db")
    migrations = [
        """
        CREATE TABLE results (
            timestamp    DATETIME,
            dataset      TEXT,
            algorithm    TEXT,
            k            INT,
            delta        REAL,
            params       JSON,
            radius       REAL,
            time_s       REAL,
            additive_violation INT,
            additional_metrics  JSON,
            hdf5_key     TEXT
        );
        """,
        """
        ALTER TABLE results ADD COLUMN timeout_s REAL;
        """
    ]
    dbver = db.execute("PRAGMA user_version").fetchone()[0]
    for i, mig in enumerate(migrations):
        ver = i + 1
        if dbver < ver:
            db.executescript(mig)
            db.execute(f"PRAGMA user_version = {ver}")

    return db


def already_run(dataset, algorithm, k, delta, attrs_dict):
    with get_db() as db:
        key = compute_key(dataset, algorithm, k, delta, attrs_dict)
        res = db.execute("""
        SELECT timestamp 
        FROM results 
        WHERE hdf5_key = :key
        """, {
            "key": key
        }).fetchone()
        return res is not None


def save_timeout(dataset, algorithm, k, delta, attrs_dict, timeout_s):
    key = compute_key(dataset, algorithm, k, delta, attrs_dict)
    with get_db() as db:
        db.execute("""
        INSERT INTO results VALUES (
            DATETIME('now'), :dataset, :algorithm, :k, :delta, :params, 
            :radius, :time_s, :additive_violation, 
            :additional_metrics, :hdf5_key, :timeout_s
        )
        """, {
            "dataset": dataset,
            "algorithm": algorithm,
            "k": k,
            "delta": delta,
            "params": json.dumps(attrs_dict, sort_keys=True),
            "timeout_s": timeout_s,
            "radius": None,
            "time_s": None,
            "additive_violation": None,
            "additional_metrics": json.dumps({}, sort_keys=True),
            "hdf5_key": key
        })


def save_result(opath, centers, assignment, dataset, algorithm, k, delta, attrs_dict, time_s, additional_metrics):
    data, colors, fairness_constraints = datasets.load(dataset, 0, delta)
    radius = assess.radius(data, centers, assignment)
    violation = assess.additive_violations(
        k, colors, assignment, fairness_constraints)
    key = write_clustering(opath, centers, assignment,
                           dataset, algorithm, k, delta, attrs_dict)
    print(f"{algorithm} on {dataset} (k={k}, delta={delta}): t={time_s}s r={radius} violation={violation}")
    with get_db() as db:
        db.execute("""
        INSERT INTO results VALUES (
            DATETIME('now'), :dataset, :algorithm, :k, :delta, :params, 
            :radius, :time_s, :additive_violation, 
            :additional_metrics, :hdf5_key, :timeout_s
        )
        """, {
            "dataset": dataset,
            "algorithm": algorithm,
            "k": k,
            "delta": delta,
            "params": json.dumps(attrs_dict, sort_keys=True),
            "radius": radius,
            "time_s": time_s,
            "timeout_s": None,
            "additive_violation": violation,
            "additional_metrics": json.dumps(additional_metrics, sort_keys=True),
            "hdf5_key": key
        })


if __name__ == "__main__":
    import kcenter
    import datasets
    k = 64
    dataset = "creditcard"
    data, colors, color_proportion = datasets.load(dataset, 0)
    greedy_centers, greedy_assignment = kcenter.greedy_minimum_maximum(data, k)
    write_clustering("results.hdf5", greedy_assignment,
                     greedy_centers, dataset, "gmm", k, {})
    centers, assignment = read_clustering(
        "results.hdf5", dataset, "gmm", k, {})
    print(centers)
    print(assignment)
