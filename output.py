import h5py
import hashlib
import os
import logging


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


def run_hash(dataset, algorithm, k, attrs_dict):
    state = hashlib.sha256()
    state.update(dataset.encode("utf8"))
    state.update(algorithm.encode("utf8"))
    state.update(bytes(str(k), encoding="utf8"))
    _hash_dict_update(state, attrs_dict)
    return state.hexdigest()


def compute_key(dataset, algorithm, k, attrs_dict):
    hash = run_hash(dataset, algorithm, k, attrs_dict)
    key = f"{dataset}/{algorithm}/{k}/{hash}"
    return key


def write_clustering(opath, centers, assignment, dataset, algorithm, k, attrs_dict):
    mode = "r+" if os.path.isfile(opath) else "w"
    with h5py.File(opath, mode) as hfp:
        key = compute_key(dataset, algorithm, k, attrs_dict)
        if key in hfp:
            logging.warn("Name already exists in result file, skipping: %s", key)
            return
        group = hfp.create_group(key)
        group["centers"] = centers
        group["assignment"] = assignment
        for k, v in attrs_dict.items():
            group.attrs[k] = v


def read_clustering(opath, dataset, algorithm, k, attrs_dict):
    with h5py.File(opath, "r") as hfp:
        key = compute_key(dataset, algorithm, k, attrs_dict)
        if key not in hfp:
            logging.error("Name missing in result file: %s", key)
            raise Exception("Name missing in result file")
        centers = hfp[f"{key}/centers"][:]
        assignment = hfp[f"{key}/assignment"][:]
        return centers, assignment



if __name__ == "__main__":
    import kcenter
    import datasets
    k = 64
    dataset = "creditcard"
    data, colors, color_proportion = datasets.load(dataset, 0)
    greedy_centers, greedy_assignment = kcenter.greedy_minimum_maximum(data, k)
    write_clustering("results.hdf5", greedy_assignment, greedy_centers, dataset, "gmm", k, {})
    centers, assignment = read_clustering("results.hdf5", dataset, "gmm", k, {})
    print(centers)
    print(assignment)

