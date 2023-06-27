# UMAP uses Numba, so here we silence warnings on which we have no control
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
import h5py
import polars as pl
import pandas as pd
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from numba.typed import List
import requests
import os
import zipfile


def download(url, local_filename=None):
    if local_filename is None:
        local_filename = url.split('/')[-1]
    if os.path.isfile(local_filename):
        return local_filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return local_filename


def write_hdf5(data, colors, encoders, fname):
    with h5py.File(fname, "w") as hfp:
        hfp["data"] = data
        hfp["data-PCA"] = PCA().fit_transform(data)
        if data.shape[0] < 40000:
            hfp["data-UMAP"] = umap.UMAP().fit_transform(data)
        else:
            warnings.warn("Dataset too large, skipping UMAP mapping")
        hfp["colors"] = colors
        for k in encoders:
            hfp["colors"].attrs[f"encoding-{k}"] = encoders[k].classes_.astype(bytes)


def creditcard():
    ofname = "data/creditcard.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    fname = download(url, "data/creditcard.zip")
    with zipfile.ZipFile(fname) as fpzip:
        with fpzip.open("default of credit card clients.xls") as fp:
            df = pd.read_excel(
                fp,
                header=1
            )
            df = pl.from_pandas(df)

    attributes = ["LIMIT_BAL","AGE","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
    colors = ["SEX", "EDUCATION", "MARRIAGE"]

    data = df.select(attributes).to_numpy()
    data = StandardScaler().fit_transform(data)

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("SEX").map(lambda c: encoders["SEX"].fit_transform(c)).explode(),
        pl.col("MARRIAGE").map(lambda c: encoders["MARRIAGE"].fit_transform(c)).explode(),
        pl.col("EDUCATION").map(lambda c: encoders["EDUCATION"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def census1990():
    ofname = "data/census1990.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://web.archive.org/web/20170711094723/https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
    fname = download(url, "data/census1990.txt")
    df = pl.read_csv(fname)
    attributes = ["dAncstry1","dAncstry2","iAvail","iCitizen","iClass","dDepart","iDisabl1","iDisabl2","iEnglish","iFeb55","iFertil","dHispanic","dHour89","dHours","iImmigr","dIncome1","dIncome2","dIncome3","dIncome4","dIncome5","dIncome6","dIncome7","dIncome8","dIndustry","iKorean","iLang1","iLooking","iMarital","iMay75880","iMeans","iMilitary","iMobility","iMobillim","dOccup","iOthrserv","iPerscare","dPOB","dPoverty","dPwgt1","iRagechld","dRearning","iRelat1","iRelat2","iRemplpar","iRiders","iRlabor","iRownchld","dRpincome","iRPOB","iRrelchld","iRspouse","iRvetserv","iSchool","iSept80","iSubfam1","iSubfam2","iTmpabsnt","dTravtime","iVietnam","dWeek89","iWork89","iWorklwk","iWWII","iYearsch","iYearwrk","dYrsserv"]
    colors = ["dAge", 'iSex']

    data = df.select(attributes).to_numpy()
    data = StandardScaler().fit_transform(data)

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("dAge").map(lambda c: encoders["dAge"].fit_transform(c)).explode(),
        pl.col("iSex").map(lambda c: encoders["iSex"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def adult():
    ofname = "data/adult.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    fname = download(url, "data/adult.zip")
    with zipfile.ZipFile(fname) as fpzip:
        with fpzip.open("adult.data") as fp:
            df = pl.read_csv(
                fp, 
                has_header=False,
                new_columns=["age", "workclass", "final-weight", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"],
                null_values="?"
            )

    colors = ["sex", "race", "marital-status"]
    attributes = ["age", "final-weight", "education-num", "capital-gain", "hours-per-week"]
    data = df.select(attributes).to_numpy()
    data = StandardScaler().fit_transform(data)

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("sex").map(lambda c: encoders["sex"].fit_transform(c)).explode(),
        pl.col("marital-status").map(lambda c: encoders["marital-status"].fit_transform(c)).explode(),
        pl.col("race").map(lambda c: encoders["race"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def diabetes():
    ofname = "data/diabetes.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130+us+hospitals+for+years+1999+2008.zip"
    fname = download(url, "data/diabetes.zip")
    with zipfile.ZipFile(fname) as fpzip:
        with fpzip.open("dataset_diabetes/diabetic_data.csv") as fp:
            df = pl.read_csv(fp, null_values="?")

    colors = ["gender", "race"]
    attributes = [
        "age",
        # "weight", # the weight column has too many missing values
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "diag_1",
        "diag_2",
        "diag_3",
        "number_diagnoses"
    ]
    df = (df
        .select(pl.col(attributes), pl.col(colors))
        .with_columns(pl.col(["age"])
                      .str.extract_all("(\\d)+")
                      .list.first()
                      .cast(pl.Int64))
        .with_columns(pl.col(["diag_1", "diag_2", "diag_3"]).cast(pl.Float64, strict=False))
        .drop_nulls()
    )
    data = df.select(attributes).to_numpy()
    data = StandardScaler().fit_transform(data)
    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("gender").map(lambda c: encoders["gender"].fit_transform(c)).explode(),
        pl.col("race").map(lambda c: encoders["race"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def _kfc_csv(url, local_fname, ofname, attributes, color_columns, sep=","):
    """
    Download and preprocess datasets available at https://github.com/FaroukY/KFC-ScalableFairClustering
    which is the repository for the paper
    "KFC: A Scalable Approximation Algorithm for k−center Fair Clustering" in NeurIPS 2020.
    """
    if os.path.isfile(ofname):
        return ofname

    fname = download(url, local_fname)
    df = pl.read_csv(fname, separator=sep)
    print(df)
    data = df.select(attributes).to_numpy()
    data = StandardScaler().fit_transform(data)
    encoders = dict((c, LabelEncoder()) for c in color_columns)
    colors = df.select([
        pl.col(column).map(lambda c: enc.fit_transform(c)).explode()
        for column, enc in encoders.items()
    ]).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def four_area():
    return _kfc_csv(
        url = "https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/4area.csv",
        local_fname = "data/4area.csv",
        ofname = "data/4area.hdf5",
        attributes = [str(c + 1) for c in range(8)],
        color_columns = ["color"]
    )


def c50():
    return _kfc_csv(
        url = "https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/c50.csv",
        local_fname = "data/c50.csv",
        ofname = "data/c50.hdf5",
        attributes = [str(c) for c in range(10)],
        color_columns = ["color"]
    )


def victorian():
    return _kfc_csv(
        url = "https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/victorian.csv",
        local_fname = "data/victorian.csv",
        ofname = "data/victorian.hdf5",
        attributes = [str(c) for c in range(10)],
        color_columns = ["color"]
    )


def bank():
    return _kfc_csv(
        url = "https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/bank_categorized.csv",
        local_fname = "data/bank_categorized.csv",
        ofname = "data/bank_categorized.hdf5",
        attributes = ["age", "balance", "duration", "job", "education", "default", "housing", "loan", "contact"],
        color_columns = ["marital"]
    )


DATASETS = {
    "adult": adult,
    "diabetes": diabetes,
    "census1990": census1990,
    "creditcard": creditcard,
    "4area": four_area,
    "reuters_50_50": c50,
    "victorian": victorian,
    "bank": bank,
}


def datasets():
    """Return all dataset names"""
    names = list(DATASETS.keys())
    return names


def load(name, color_idx, delta=0.0, prefix=None):
    fname = DATASETS[name]()
    print("Opening", fname)
    with h5py.File(fname, "r") as hfp:
        if prefix is None:
            data = hfp["data"][:]
            colors = hfp["colors"][:, color_idx]
        else:
            data = hfp["data"][:prefix, :]
            colors = hfp["colors"][:prefix, color_idx]
    unique_colors, color_counts = np.unique(colors, return_counts=True)
    color_proportion = color_counts / np.sum(color_counts)
    fairness_constraints = List([
        (p * (1-delta), p / (1-delta))
        for p in color_proportion
    ])
    return data, colors, fairness_constraints


def load_pca2(name):
    fname = DATASETS[name]()
    with h5py.File(fname, "r") as hfp:
        data = hfp["data-PCA"][:,:2]
    return data

def load_umap(name):
    fname = DATASETS[name]()
    with h5py.File(fname, "r") as hfp:
        if "data-UMAP" not in hfp:
            return None
        data = hfp["data-UMAP"][:]
    return data


if __name__ == "__main__":
    for dataset in DATASETS:
        print("Preprocessing", dataset)
        preprocessing = DATASETS[dataset]
        preprocessing()

