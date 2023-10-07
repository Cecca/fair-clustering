# UMAP uses Numba, so here we silence warnings on which we have no control
import logging
import zipfile
import os
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import polars as pl
import h5py


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
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
    import warnings
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    import umap

    assert not np.any(np.isnan(data))
    with h5py.File(fname, "w") as hfp:
        hfp["data"] = data.astype(np.float64)
        hfp["data-PCA"] = PCA().fit_transform(data)
        if data.shape[0] < 40000:
            hfp["data-UMAP"] = umap.UMAP().fit_transform(data)
        else:
            warnings.warn("Dataset too large, skipping UMAP mapping")
        hfp["colors"] = colors
        for k in encoders:
            if hasattr(encoders[k], "classes_"):
                clss = encoders[k].classes_.astype(bytes)
            else:
                clss = encoders[k]
            hfp["colors"].attrs[f"encoding-{k}"] = clss


def standardize(dataset_fn):
    def inner():
        data_name = dataset_fn()
        oname = data_name.replace(".hdf5", ".standardized.hdf5")
        if os.path.isfile(oname):
            return oname
        with h5py.File(data_name, "r") as hfp:
            data = hfp["data"][:]
            colors = hfp["colors"][:]
            encoders = dict((k, v) for k, v in hfp["colors"].attrs.items()
                            if k.startswith("encoding"))
        data = StandardScaler().fit_transform(data)
        write_hdf5(data, colors, encoders, oname)
        return oname
    return inner


def random_dbg():
    """A small random dataset for debugging purposes"""
    fname = "data/random.hdf5"
    if os.path.isfile(fname):
        return fname
    n = 10
    data = np.random.standard_normal((n, 2))
    colors = np.random.random_integers(0, 1, (n, 1))
    write_hdf5(data, colors, {}, fname)
    return fname


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

    attributes = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
                  "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    colors = ["SEX", "EDUCATION", "MARRIAGE"]

    data = df.select(attributes).to_numpy()
    # data = StandardScaler().fit_transform(data)

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("SEX").map(
            lambda c: encoders["SEX"].fit_transform(c)).explode(),
        pl.col("MARRIAGE").map(
            lambda c: encoders["MARRIAGE"].fit_transform(c)).explode(),
        pl.col("EDUCATION").map(
            lambda c: encoders["EDUCATION"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def census1990():
    ofname = "data/census1990.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://web.archive.org/web/20170711094723/https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
    fname = download(url, "data/census1990.txt")
    df = pl.read_csv(fname).drop_nulls()
    attributes = ["dAncstry1", "dAncstry2", "iAvail", "iCitizen", "iClass", "dDepart",
                  "iDisabl1", "iDisabl2", "iEnglish", "iFeb55", "iFertil", "dHispanic",
                  "dHour89", "dHours", "iImmigr", "dIncome1", "dIncome2", "dIncome3", "dIncome4", "dIncome5", "dIncome6", "dIncome7", "dIncome8", "dIndustry", "iKorean", "iLang1", "iLooking", "iMarital", "iMay75880", "iMeans", "iMilitary", "iMobility", "iMobillim", "dOccup", "iOthrserv", "iPerscare", "dPOB", "dPoverty", "dPwgt1", "iRagechld", "dRearning", "iRelat1", "iRelat2", "iRemplpar", "iRiders", "iRlabor", "iRownchld", "dRpincome", "iRPOB", "iRrelchld", "iRspouse", "iRvetserv", "iSchool", "iSept80",
                  "iSubfam1", "iSubfam2", "iTmpabsnt", "dTravtime", "iVietnam", "dWeek89",
                  "iWork89", "iWorklwk", "iWWII", "iYearsch", "iYearwrk", "dYrsserv"]
    colors = ["dAge", 'iSex']

    data = df.select(attributes).to_numpy()

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("iSex").map(
            lambda c: encoders["iSex"].fit_transform(c)).explode(),
        pl.col("dAge").map(
            lambda c: encoders["dAge"].fit_transform(c)).explode()
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def census1990_age():
    ofname = "data/census1990-age.hdf5"
    if os.path.isfile(ofname):
        return ofname

    url = "https://web.archive.org/web/20170711094723/https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
    fname = download(url, "data/census1990.txt")
    df = pl.read_csv(fname).drop_nulls()
    attributes = ["dAncstry1", "dAncstry2", "iAvail", "iCitizen", "iClass", "dDepart",
                  "iDisabl1", "iDisabl2", "iEnglish", "iFeb55", "iFertil", "dHispanic",
                  "dHour89", "dHours", "iImmigr", "dIncome1", "dIncome2", "dIncome3",
                  "dIncome4", "dIncome5", "dIncome6", "dIncome7", "dIncome8", "dIndustry",
                  "iKorean", "iLang1", "iLooking", "iMarital", "iMay75880", "iMeans",
                  "iMilitary", "iMobility", "iMobillim", "dOccup", "iOthrserv", "iPerscare",
                  "dPOB", "dPoverty", "dPwgt1", "iRagechld", "dRearning", "iRelat1", "iRelat2",
                  "iRemplpar", "iRiders", "iRlabor", "iRownchld", "dRpincome", "iRPOB",
                  "iRrelchld", "iRspouse", "iRvetserv", "iSchool", "iSept80",
                  "iSubfam1", "iSubfam2", "iTmpabsnt", "dTravtime", "iVietnam", "dWeek89",
                  "iWork89", "iWorklwk", "iWWII", "iYearsch", "iYearwrk", "dYrsserv"]
    colors = ["dAge", 'iSex']

    data = df.select(attributes).to_numpy()

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("dAge").map(
            lambda c: encoders["dAge"].fit_transform(c)).explode()
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
                new_columns=["age", "workclass", "final-weight", "education", "education-num", "marital-status", "occupation",
                             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"],
                null_values="?"
            )

    colors = ["sex", "race", "marital-status"]
    attributes = ["age", "final-weight", "education-num",
                  "capital-gain", "hours-per-week"]
    data = df.select(attributes).to_numpy()
    # data = StandardScaler().fit_transform(data)

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("sex").map(
            lambda c: encoders["sex"].fit_transform(c)).explode(),
        pl.col(
            "marital-status").map(lambda c: encoders["marital-status"].fit_transform(c)).explode(),
        pl.col("race").map(
            lambda c: encoders["race"].fit_transform(c)).explode()
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
    # data = StandardScaler().fit_transform(data)
    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("gender").map(
            lambda c: encoders["gender"].fit_transform(c)).explode(),
        pl.col("race").map(
            lambda c: encoders["race"].fit_transform(c)).explode()
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
    # data = StandardScaler().fit_transform(data)
    encoders = dict((c, LabelEncoder()) for c in color_columns)
    colors = df.select([
        pl.col(column).map(lambda c: enc.fit_transform(c)).explode()
        for column, enc in encoders.items()
    ]).to_numpy()
    write_hdf5(data, colors, encoders, ofname)
    return ofname


def four_area():
    return _kfc_csv(
        url="https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/4area.csv",
        local_fname="data/4area.csv",
        ofname="data/4area.hdf5",
        attributes=[str(c + 1) for c in range(8)],
        color_columns=["color"]
    )


def c50():
    return _kfc_csv(
        url="https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/c50.csv",
        local_fname="data/c50.csv",
        ofname="data/c50.hdf5",
        attributes=[str(c) for c in range(10)],
        color_columns=["color"]
    )


def victorian():
    return _kfc_csv(
        url="https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/victorian.csv",
        local_fname="data/victorian.csv",
        ofname="data/victorian.hdf5",
        attributes=[str(c) for c in range(10)],
        color_columns=["color"]
    )


def bank():
    return _kfc_csv(
        url="https://github.com/FaroukY/KFC-ScalableFairClustering/raw/main/data/bank_categorized.csv",
        local_fname="data/bank_categorized.csv",
        ofname="data/bank_categorized.hdf5",
        attributes=["age", "balance", "duration", "job",
                    "education", "default", "housing", "loan", "contact"],
        color_columns=["marital"]
    )


def hmda():
    """
    Home Mortgage Disclosure Act
    ============================

    Source: https://ffiec.cfpb.gov/data-publication/modified-lar/2022
    """
    ofile = "data/hmda.hdf5"
    if os.path.isfile(ofile):
        return ofile
    url = "https://s3.amazonaws.com/cfpb-hmda-public/prod/dynamic-data/combined-mlar/2022/header/2022_combined_mlar_header.zip"
    local = download(url, "data/hmda.zip")
    unzipped = "data/2022_combined_mlar_header.txt"
    if not os.path.isfile(unzipped):
        with zipfile.ZipFile(local) as fpzip:
            fpzip.extract("2022_combined_mlar_header.txt", "data")

    attr_cols = ["loan_amount", "total_loan_costs", "origination_charges", "discount_points",
                 "lender_credits", "interest_rate", "combined_loan_to_value_ratio", "loan_term"]
    df = pl.scan_csv(unzipped, separator="|",
                     null_values=["NA", ""],
                     dtypes=dict((c, "str") for c in attr_cols),
                     ignore_errors=False).with_columns(
        pl.col(c).str.replace("Exempt", "0").cast(
            "f64").fill_null(pl.lit(0)).alias(c)
        for c in attr_cols
    )
    color_col = "applicant_race_1"

    df = df.select([color_col] + attr_cols).drop_nulls().collect()

    # Remove blatant outliers
    for attr in attr_cols:
        df = df.filter(
            pl.col(attr).rank(descending=True) >= 10000
        )
    data = df.select(attr_cols).to_numpy()
    print(data.shape)

    encoders = dict((c, LabelEncoder()) for c in [color_col])
    colors = df.select(
        pl.col("applicant_race_1").map(
            lambda c: encoders["applicant_race_1"].fit_transform(c)).explode(),
    ).to_numpy()
    write_hdf5(data, colors, encoders, ofile)
    os.remove(unzipped)
    return ofile


def athlete():
    ofile = "data/athlete.hdf5"
    if os.path.isfile(ofile):
        return ofile
    url = "https://github.com/rgriff23/Olympic_history/raw/master/data/athlete_events.csv"
    local = download(url, "data/athlete.csv")

    attributes = ["Age", "Height", "Weight"]
    colors = ["Sex"]

    df = pl.read_csv(local, null_values=["", "NA"]).select(
        attributes + colors).drop_nulls()

    data = df.select(attributes).to_numpy()

    encoders = dict((c, LabelEncoder()) for c in colors)
    colors = df.select(
        pl.col("Sex").map(
            lambda c: encoders["Sex"].fit_transform(c)).explode()
    ).to_numpy()

    write_hdf5(data, colors, encoders, ofile)

    return ofile


DATASETS = {
    "adult": adult,
    "athlete": athlete,
    "diabetes": diabetes,
    "census1990": census1990,
    "census1990_age": census1990_age,
    "creditcard": creditcard,
    "4area": four_area,
    "reuter_50_50": c50,
    "victorian": victorian,
    "bank": bank,
    "hmda": hmda,
    "random_dbg": random_dbg
}

# Add standardized datasets
# for dataset in list(DATASETS.keys()):
#     DATASETS[f"{dataset}-std"] = standardize(DATASETS[dataset])


def datasets():
    """Return all dataset names"""
    names = [k for k in DATASETS.keys() if k != "random_dbg" and k !=
             "random_dbg-std"]
    return names


def load(name, color_idx, delta=0.0, prefix=None, shuffle_seed=None):
    fname = DATASETS[name]()
    logging.debug("Opening %s", fname)
    with h5py.File(fname, "r") as hfp:
        if prefix is None:
            data = hfp["data"][:]
            colors = hfp["colors"][:, color_idx]
        else:
            data = hfp["data"][:prefix, :]
            colors = hfp["colors"][:prefix, color_idx]
    unique_colors, color_counts = np.unique(colors, return_counts=True)
    color_proportion = color_counts / np.sum(color_counts)
    fairness_constraints = [
        (p * (1-delta), p / (1-delta))
        for p in color_proportion
    ]
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        selector = np.arange(data.shape[0])
        rng.shuffle(selector)
        data = data[selector]
        colors = colors[selector]
    return data, colors, fairness_constraints


def dataset_ncolors(name):
    fname = DATASETS[name]()
    logging.debug("Opening %s", fname)
    with h5py.File(fname, "r") as hfp:
        return int(hfp["colors"][:].max() + 1)


def dataset_size(name):
    fname = DATASETS[name]()
    logging.debug("Opening %s", fname)
    with h5py.File(fname, "r") as hfp:
        return hfp["data"].shape


def load_pca2(name):
    fname = DATASETS[name]()
    with h5py.File(fname, "r") as hfp:
        data = hfp["data-PCA"][:, :2]
        colors = hfp["colors"][:, 0]
    return data, colors


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
