# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:48:30 2022

@author: fhu14
"""
"""
Starter code for generating/reproducing the heatmap by comparing different methods.
For more complete documentation, refer to the functions defined in the module
ani1_interface
"""
# %% Imports, definitions

import itertools
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ani1_interface import get_ani1data
import seaborn as sns
import matplotlib.pyplot as plt
import re

from tqdm import tqdm

Array = np.ndarray


# %% Code behind


def build_XX_matrix(dataset: List[Dict], allowed_Zs: List[int]) -> Array:
    nmol = len(dataset)
    XX = np.zeros([nmol, len(allowed_Zs) + 1])
    iZ = {x: i for i, x in enumerate(allowed_Zs)}

    for imol, molecule in enumerate(dataset):
        Zc = Counter(molecule["atomic_numbers"])
        for Z, count in Zc.items():
            # td: The XX matrix is uniquely determined by `dataset` and
            # could be cached
            XX[imol, iZ[Z]] = count
            XX[imol, len(allowed_Zs)] = 1.0

    return XX


def fit_linear_ref_ener(
    dataset: List[Dict],
    target1: str,
    target2: str,
    allowed_Zs: List[int],
    XX: Optional[Array] = None,
) -> Array:
    r"""Fits a linear reference energy model between the DFTB+ method and some
        energy target

    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries that have had the
            DFTB+ results added to them.
        target1 (str): The starting point energy target
        target2 (str): The second energy target that you need to correct for
        allowed_Zs (List[int]): The allowed atoms in the molecules

    Returns:
        coefs (Array): The coefficients of the reference energy
        XX (Array): 2D matrix in the number of atoms
        method1_mat (Array): The reference energy of the DFTB+ method
        method2_mat (Array): The reference energy of the target
        XX (Array): Per-molecule atomic frequency matrix

    Notes: The reference energy corrects the magnitude between two methods
        in the following way:

        E_2 = E_1 + sum_z N_z * C_z + C_0

        where N_z is the number of times atom z occurs within the molecule and
        C_z is the coefficient for the given molecule. This is accomplished by solving
        a least squares problem.

        The reference energy vector is generated through a matrix multiply.
        Suppose that E_1 is the vector of energies for the molecules in the
        given dataset. The corrected energies, E_corrected, is generated as follows:

        E_corrected = E_1 + (XX @ coefs)

        where XX and coefs are the output of this function.
    """
    nmol = len(dataset)

    if XX is None:
        XX = build_XX_matrix(dataset, allowed_Zs)
    else:
        expected = (nmol, len(allowed_Zs) + 1)
        if XX.shape != expected:
            raise ValueError(
                f"Expected XX to have shape {expected}, but got {XX.shape}"
            )

    method1_mat = np.zeros([nmol])
    method2_mat = np.zeros([nmol])

    for imol, molecule in enumerate(dataset):
        method1_mat[imol] = molecule["targets"][target1]
        method2_mat[imol] = molecule["targets"][target2]

    yy = method2_mat - method1_mat

    lsq_res = np.linalg.lstsq(XX, yy, rcond=None)
    coefs = lsq_res[0]
    return coefs, XX, method1_mat, method2_mat


"""
Consider the following workflow
    1) Use get_ani1data and the ani1 h5 data file (found on Canvas) to generate
       a series of  molecules from the overall ANI-1ccx dataset
    2) Save the molecules using pickle
    3) Calculate differences and save those too (This will involve using the
       above function for generating linear reference energy parameters)
    4) Generate the heatmap using seaborn
    5) Think about outlier rejection

I recommend that you guys work collaboratively on this, whether that is dividing
the work into different functions or programming together. Again, the method of
working is up to your group.
"""


def get_ani1data_cached(
    ani1_path: str,
    molecules_path: str,
    allowed_Z: List[int],
    heavy_atoms: List[int],
    max_config: int,
    target: Dict[str, str],
    **kwargs,
) -> List[Dict]:
    r"""Loads the ani1 data file and returns the molecules in the file

    Arguments:
        ani1_path (str): The path to the ani1 data file
        molecules_path (str): The path to the pickled molecules file
        allowed_Z (List[int]): Include only molecules whose elements are in
            this list
        heavy_atoms (List[int]): Include only molecules for which the number
            of heavy atoms is in this list
        max_config (int): Maximum number of configurations included for each
            molecule.
        target (Dict[str,str]): entries specify the targets to extract
            key: target_name name assigned to the target
            value: key that the ANI-1 file assigns to this target
    Returns:
        molecules (List[Dict]): The list of molecule dictionaries
    """

    if not os.path.exists(molecules_path):
        molecules = get_ani1data(
            allowed_Z=allowed_Z,
            heavy_atoms=heavy_atoms,
            max_config=max_config,
            target=target,
            ani1_path=ani1_path,
            **kwargs,
        )
        with open(molecules_path, "wb") as f:
            pickle.dump(molecules, f)
    else:
        with open(molecules_path, "rb") as f:
            molecules = pickle.load(f)


def calc_resid(
    molecules: List[Dict],
    target: str,
    allowed_Z: List[int],
    show_progress: bool = False,
    XX: Optional[Array] = None,
) -> Dict[Tuple[str, str], Array]:
    r"""calculates residuals of the ani1 data set

        Arguments:
            molecules List[Dict]: From ANI-1 dataset
            allowed_Z List[int]: The allowed atoms in the molecules
            target (str): energy targets
            show_progress (bool): Show TQDM progress bar
            XX (Optional[Array]): precomputed array to replace molecules

        Returns:
            resid_matrix Dict: matrix of the residuals between two methods

        Notes:
            Result is converted to hartrees
        """
    n_targets = len(target.keys())

    target_keys = list(target.keys())

    # List indices
    target_idx_pairs = list(itertools.combinations(range(n_targets), 2))

    conversion = 627.50961

    if show_progress:
        target_idx_pairs = tqdm(target_idx_pairs)

    resid_matrix = {}

    for (idx_1, idx_2) in target_idx_pairs:
        coefs, XX, method1_mat, method2_mat = fit_linear_ref_ener(
            molecules, target_keys[idx_1], target_keys[idx_2], allowed_Z, XX=XX
        )

        resid = method2_mat - (method1_mat + (XX @ coefs))
        resid = resid * conversion
        target_1_name = target_keys[idx_1]
        target_2_name = target_keys[idx_2]
        resid_matrix[(target_1_name, target_2_name)] = resid

    return resid_matrix

def create_heatmap(
    target: str,
    data_matrix: Optional[List[Dict]] = None,
    molecules: Optional[List[Dict]] = None,
    allowed_Z: Optional[List[int]] = None,
    plot_args: Optional[Dict] = None,
    show_progress: bool = False,
    XX: Optional[Array] = None,

) -> plt.Axes:
    """Creates a heatmap of the MAE between methods.

    Args:
        target (str): List of method IDs to compare
        data_matrix (Optional[List[Dict]]): residual matrix
        molecules (Optional(List[Dict])): From ANI-1 dataset
        allowed_Z (Optional(List[int])): The allowed atoms in the molecules
        plot_args (Optional[Dict]): Arguments to pass to seaborn heatmap
        show_progress (bool): Show TQDM progress bar
        XX (Optional[Array]): precomputed array to replace molecules

    Returns:
        plt.Axes: Matplotlib axes object

    Notes:
        Refactored to take in the residual matrix by default
    """

    if data_matrix is None and molecules is None:
        raise ValueError("One of data_matrix or molecules must be provided")

    target_values = list(target.values())

    n_targets = len(target.keys())
    mae_matrix = np.zeros((n_targets, n_targets))

    # List indices
    target_idx_pairs = list(itertools.combinations(range(n_targets), 2))

    if show_progress:
        target_idx_pairs = tqdm(target_idx_pairs)

    if data_matrix is None:
        data_matrix = calc_resid(molecules, target, allowed_Z, show_progress=show_progress, XX=XX)

    for (idx_1, idx_2), (ind_1, ind_2) in itertools.zip_longest(data_matrix, target_idx_pairs):
        resid = data_matrix[idx_1, idx_2]
        mae_matrix[ind_2, ind_1] = np.mean(np.abs(resid))

    # Mask for seaborn heatmap, to remove the upper triangular portion,
    # but including the main diagonal
    mask = np.triu(np.ones_like(mae_matrix), k=1)
    ax = sns.heatmap(
        mae_matrix,
        annot=True,
        fmt=".1f",
        xticklabels=target_values,
        yticklabels=target_values,
        mask=mask,
        **(plot_args or {}),
    )

    return ax


def filter_outliers(
    data_matrix: Dict[Tuple[str, str], Array]
) -> Dict[Tuple[str, str], Array]:
    """Filters outliers from each element in the dataset

    Arguments:
        data_matrix (Dict): dictionary with the mean absolute error

    Returns:
        filtered_dict (Dict): matrix with no outliers

    Notes: Using the IQR to calc outliers
    """
    filtered_dict = {}
    for (target1, target2), resid in data_matrix.items():
        q1, q3 = np.percentile(resid, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr

        filtered_element = resid[resid < upper_bound]
        filtered_element = filtered_element[filtered_element > lower_bound]
        filtered_dict[(target1, target2)] = filtered_element

    return filtered_dict


# Functions for heavy atom residual analysis


def num_heavy_atoms(name: str) -> int:
    matches = re.findall(r"[A-Z]\d+", name)
    num_heavy = sum(int(m[1:]) for m in matches if not m.startswith("H"))
    return num_heavy


def get_residuals_by_num_heavy_atoms(
    molecules: List[Dict], residuals: Array, heavy_atoms: list[int]
) -> Dict:

    molecules_by_heavy_atoms = {x: [] for x in heavy_atoms}

    for i, molecule in enumerate(molecules):
        num_heavy = num_heavy_atoms(molecule["name"])

        resid_for_molecule = residuals[i]
        molecules_by_heavy_atoms[num_heavy].append(resid_for_molecule)

    return molecules_by_heavy_atoms


def rmse(y: Array, y_pred: Optional[Array] = None) -> float:
    """Calculate the root mean squared error between y and y_pred.

    If y_pred is not provided, y is treated as the residual vector.
    """

    if y_pred is None:
        rmse = np.sqrt(np.mean(np.power(y, 2)))
    else:
        rmse = np.sqrt(np.mean(np.power((y - y_pred), 2)))

    return rmse


def compute_rmse_by_num_heavy_atoms(
    molecules: List[Dict],
    resid: Dict[Tuple[str, str], Array],
    heavy_atoms: list[int],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calculates the heavy-atom conditional RMSE for each method-method combination.

    Args:
        molecules (List[Dict]): List of molecules dictionaries from ANI-1 data.
        resid (Dict): Dictionary of residual vectors for each method-method combination.
        heavy_atoms (list[int]): List of allowed heavy atom numbers.
        show_progress (bool): Whether to display the TQDM progress bar.

    Returns:
        pd.DataFrame: Dataframe with the RMSE conditional on heavy atoms for
        each method-method combination. Also includes STD, which is the standard
        deviation of the residual vector, and n, which is the number of residuals
        used in the calculation.
    """

    # TODO: This function can be simplified through vectorization, but is
    # written this way because of how the method-method residuals are
    # stored in the resid dictionary.
    dataframes = []

    for method_pair in tqdm(resid, desc="Computing RMSE", disable=not show_progress):
        resids_by_heaviness = get_residuals_by_num_heavy_atoms(
            molecules, resid[method_pair], heavy_atoms
        )

        # We have about 389 molecules with 1 heavy atom
        rmse_vals = []
        rmse_nh_vals = []
        sd_vals = []
        num_molecules = []
        for num_heavy_atoms in heavy_atoms:
            # resids_by_heaviness[num_heavy_atoms] is a dictionary,
            # mapping number of heavy atoms to the subset of the molecule-level
            # residuals corresponding to the molecules with num_heavy_atoms
            # (for a given method-method pair)
            rmse_val = rmse(resids_by_heaviness[num_heavy_atoms])
            rmse_vals.append(rmse_val)
            rmse_nh_vals.append(rmse_val / num_heavy_atoms**0.5)

            sd_vals.append(np.std(resids_by_heaviness[num_heavy_atoms]))
            num_molecules.append(len(resids_by_heaviness[num_heavy_atoms]))

        method_pair_rmse_df = pd.DataFrame(
            {
                "RMSE": rmse_vals,
                "RMSE / sqrt(nh)": rmse_nh_vals,
                "Heavy Atoms": heavy_atoms,
                "Method Pair": [method_pair] * len(heavy_atoms),
                "STD": sd_vals,
                "n": num_molecules,
            }
        )

        dataframes.append(method_pair_rmse_df)

    rmse_df = pd.concat(dataframes)
    return rmse_df


def plot_rmse_by_num_heavy_atoms(
    rmse_df: pd.DataFrame, method_id_to_name: Optional[Dict[str, str]] = None
) -> None:
    """Plots the RMSE conditional on heavy atoms for each method-method combination.

    Args:
        rmse_df (pd.DataFrame): Dataframe with the RMSE conditional on heavy atoms for
        each method-method combination.
    """

    for (method1, method2), group in rmse_df.groupby("Method Pair"):
        if method_id_to_name is not None:
            method1_full_name = method_id_to_name[method1]
            method2_full_name = method_id_to_name[method2]

        title = f"RMSE vs. # of Heavy Atoms ({method1_full_name} - {method2_full_name})"
        # group.plot(x="Heavy Atoms", y="RMSE", title=title)
        group.set_index("Heavy Atoms")[["RMSE", "RMSE / sqrt(nh)"]].plot(title=title)
        # plt.errorbar(
        #     x=group["Heavy Atoms"],
        #     y=group["RMSE"],
        #     yerr=group["STD"] / (group["n"] ** 0.5),
        # )
        #
        # plt.title(title)
        plt.show()


def isin_tuple_series(values: Any, tuple_col: pd.Series) -> pd.Series:
    if isinstance(values, str):
        values = [values]

    return tuple_col.apply(lambda x: any(val in x for val in values))


# %% Initialize Data

# https://drive.google.com/file/d/1SP8SX0v5d1UJAX69GpMV-JtjfUSnf-QB
ani1_path = "./ANI-1ccx_clean_fullentry.h5"
# molecules_path = "./ani1-extracted.pkl"

ani1_config = {
    "allowed_Z": [1, 6, 7, 8],
    "heavy_atoms": list(range(1, 9)),
    "max_config": 1_000_000,
    "target": {
        "dt": "dftb.energy",  # Dftb Total
        "pt": "dftb_plus.energy",  # dftb Plus Total
        "hd": "hf_dz.energy",  # Hf Dz
        "ht": "hf_tz.energy",
        "hq": "hf_qz.energy",
        "wd": "wb97x_dz.energy",  # Wb97x Dz
        "wt": "wb97x_tz.energy",
        "md": "mp2_dz.energy",  # Mp2 Dz
        "mt": "mp2_tz.energy",
        "mq": "mp2_qz.energy",
        "td": "tpno_ccsd(t)_dz.energy",  # Tpno Dz
        "nd": "npno_ccsd(t)_dz.energy",  # Npno Dz
        "nt": "npno_ccsd(t)_tz.energy",
        "cc": "ccsd(t)_cbs.energy",
    },
}

molecules = get_ani1data(
    allowed_Z=ani1_config["allowed_Z"],
    heavy_atoms=ani1_config["heavy_atoms"],
    max_config=ani1_config["max_config"],
    target=ani1_config["target"],
    ani1_path=ani1_path,
)

# This will read the molecules pickle
# molecules = get_ani1data_cached(
#     ani1_path,
#     molecules_path,
#     allowed_Z=list(range(1, 10)),
#     heavy_atoms=list(range(1, 10)),
#     max_config=10,
#     target=target,
#     ani1_path=ani1_path,
# )

# Some of the molecules have NaN values for some of the targets.
# This is a problem for the OLS solver, so we drop these molecules.
molecules = [m for m in molecules if not np.isnan(list(m["targets"].values())).any()]

# Precompute the XX matrix for residual calculation -- not required
XX = build_XX_matrix(molecules, ani1_config["allowed_Z"])


# Calculate the residual vector for each method-method combination
resid = calc_resid(
    molecules,
    ani1_config["target"],
    ani1_config["allowed_Z"],
    XX=XX,
    show_progress=True,
)

# %% Original Data Visualizations

# Create a heatmap of the MAE between methods
fig, ax = plt.subplots(figsize=(16, 15))
create_heatmap(
    ani1_config["target"],
    data_matrix=resid,
    show_progress=True
)

plt.show()

# Original data boxplot
oriboxfig = plt.figure(figsize=(10, 10))
data = list(resid.values())
labels = list(resid.keys())
plt.boxplot(data, labels=labels)
plt.show()

# %% Filtering Data Visualizations

filtered_data = filter_outliers(resid)
# Filtered data boxplot

boxfig = plt.figure(figsize=(10, 10))
plt.subplots(figsize=(15, 15))
boxplot_data = list(filtered_data.values())
boxplot_labels = list(filtered_data.keys())
plt.boxplot(boxplot_data, labels=boxplot_labels)
plt.show()

# Heatmap of number of outliers
# Get number of outliers
n_outliers = {}
for (target_1, target_2) in resid:
    n_outliers[target_1, target_2] = len(resid[target_1, target_2]) - len(filtered_data[target_1, target_2])
# Plot
outlier_map = plt.subplots(figsize=(16, 15))
create_heatmap(
    ani1_config["target"],
    data_matrix=n_outliers,
    show_progress=True
)
# okay to use create_heatmap since mean of 1 number is just the number
plt.show()

# Filtered Data Heatmap
fig2, ax2 = plt.subplots(figsize=(16, 15))
create_heatmap(
    ani1_config["target"],
    data_matrix=filtered_data,
    show_progress=True
)

plt.show()

# %% RMSE

# rmse_df = compute_rmse_by_num_heavy_atoms(molecules, resid, ani1_config["heavy_atoms"])

# Will produce 91 plots, one for each method-method combination
# plot_rmse_by_num_heavy_atoms(rmse_df)

# Will produce a plot for each dftb-method combination
# plot_rmse_by_num_heavy_atoms(
#     rmse_df[isin_tuple_series("dt", rmse_df["Method Pair"])],
#     method_id_to_name=ani1_config["target"],
# )

