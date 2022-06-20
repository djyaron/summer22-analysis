# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:48:30 2022

@author: fhu14
"""
from pandas import DataFrame, Series

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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ani1_interface import get_ani1data
import seaborn as sns
import matplotlib.pyplot as plt
import re

from tqdm import tqdm

from copy import copy
from typing import Optional

Array = np.ndarray

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
        target: str = ani1_config["target"],
        allowed_Z: List[int] = ani1_config["allowed_Z"],
        show_progress: bool = True,
        XX: Optional[Array] = None,
        as_dataframe: bool = False,
) -> Union[Dict, pd.DataFrame]:
    r"""calculates residuals of the ani1 data set

    Arguments:
        molecules (List[Dict]): From ANI-1 dataset
        allowed_Z (List[int]): The allowed atoms in the molecules
        target (str): energy targets
        show_progress (bool): Show TQDM progress bar
        XX (Optional[Array]): precomputed array to replace molecules

    Returns:
        resid_matrix Dict: matrix of the residuals between two methods

    Notes:
        Result is converted to hartrees
    """
    conversion = 627.50961
    n_targets = len(target.keys())
    target_keys = list(target.keys())

    # List indices
    target_idx_pairs = list(itertools.combinations(range(n_targets), 2))

    if show_progress:
        target_idx_pairs = tqdm(target_idx_pairs, desc="Calculating residuals")

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

    if not as_dataframe:
        return resid_matrix
    else:
        result = pd.DataFrame.from_records(resid_matrix)
        result["name"] = [m["name"] for m in molecules]
        result["iconfig"] = [m["iconfig"] for m in molecules]
        result = result.set_index(["name", "iconfig"])
        return result


def create_heatmap(
        target: str,
        title: str,
        data_matrix: Optional[List[Dict]] = None,
        molecules: Optional[List[Dict]] = None,
        allowed_Z: Optional[List[int]] = None,
        plot_args: Optional[Dict] = None,
        show_progress: bool = False,
        XX: Optional[Array] = None,
):
    """Creates a heatmap of the MAE between methods.

    Args:
        target (str): List of method IDs to compare
        title (str): Title of heatmap
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
        data_matrix = calc_resid(
            molecules, target, allowed_Z, show_progress=show_progress, XX=XX
        )

    for (idx_1, idx_2), (ind_1, ind_2) in itertools.zip_longest(
            data_matrix, target_idx_pairs
    ):
        resid = data_matrix[idx_1, idx_2]
        mae_matrix[ind_2, ind_1] = np.mean(np.abs(resid))

    # Mask for seaborn heatmap, to remove the upper triangular portion,
    # but including the main diagonal
    fig, ax = plt.subplots(figsize=(16, 15))
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
    plt.title(title)
    plt.show()


def filter_outliers(
        data_matrix: Optional[Dict[Tuple[str, str], Array]] = None,
        dataframe: Optional[Union[DataFrame, Series]] = None,
        q_lower: float = 0.25, q_upper: float = 0.75
) -> Any:
    """Filters outliers from each element in the dataset

    Arguments:
        data_matrix (Optional(Dict)): dictionary with the mean absolute error
        dataframe (Optional[Union[DataFrame, Series]]): dataframe from molecules
        q_lower (float): lower quantile
        q_upper (float): upper quantile

    Returns:
        filtered_dict (Dict): matrix with no outliers
        dataframe (Union[DataFrame, Series]): dataframe with ref
        energies replaced with bool of whether it was an outlier or not

    Notes: Using the IQR to calc outliers
    """

    if data_matrix is not None:
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

    elif dataframe is not None:
        return (dataframe < dataframe.quantile(q_lower)) | (dataframe > dataframe.quantile(q_upper))
    else:
        raise ValueError("One of data_matrix or dataframe must be provided")


def is_outlier(
        x: Union[DataFrame, Series], q_lower: float = 0.25, q_upper: float = 0.75
) -> Union[DataFrame, Series]:
    return (x < x.quantile(q_lower)) | (x > x.quantile(q_upper))


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
            rmse_nh_vals.append(rmse_val / num_heavy_atoms ** 0.5)

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


def create_boxplot(
        boxplot_data: Dict,
        title: str,
        method: Optional[str] = None,
        plot_args: Optional[Dict] = None,
):
    oriboxfig = plt.figure(figsize=(10, 10))
    if method is not None:
        boxplot_data = {
            key: value for key, value in boxplot_data.items() if method in key
        }
        if not boxplot_data:
            raise ValueError("Method not found in boxplot_data")

    data = list(boxplot_data.values())
    labels = list(boxplot_data.keys())
    plt.boxplot(data, labels=labels, **(plot_args or {}))
    plt.title(f"{title} for {method}")
    plt.xticks(rotation=90)
    plt.show()


def create_histogram(
        data: DataFrame,
        plot_args: Optional[Dict] = None
):
    """Filters outliers from each element in the dataset

        Arguments:
            data (DataFrame): FILTERED data dataframe--must already count the number of outliers
            plot_args (Optional[Dict]): additional args for the histogram


        Returns:
            Nothing

        Notes: Using the IQR to calc outliers
        """
    for index in data.index:
        plt.figure(figsize=(10, 10))
        plt.hist(data.loc[f"{index}"], **(plot_args or {}))
        plt.title(f"{index} Frequency")
        plt.show()


def unnest_dictionary(
        data: dict, key: str, prefix: str = "", inplace: bool = False
) -> Optional[dict]:
    """Insert the keys of a sub-dictionary into data dictionary.

    Args:
        data (dict): Main dictionary to unnest.
        key (str): The key of the sub-dictionary to unnest.
        prefix (str, optional): String value to prefix the new keys with. Defaults to "".
        inplace (bool, optional): Modify the dictionary in place if True, else return a copy. Defaults to False.

    Returns:
        Optional[dict]: The modified dictionary if inplace is False, else None.
    """
    value = data.get(key)

    # Do nothing if the key does not exist or the value is not a dictionary
    if not isinstance(value, dict):
        return data

    if not inplace:
        data = copy(data)

    # Remove the key from the dictionary
    del data[key]

    for k, v in value.items():
        data[f"{prefix}{k}"] = v

    return None if inplace else data


def convert_ani1_data_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Converts ANI1 data to a dataframe.

    Args:
        data (List[Dict]): List of dictionaries containing ANI1 data.

                    'name': str with name ANI1 assigns to this molecule type
                    'iconfig': int with number ANI1 assignes to this structure
                    'atomic_numbers': List of Zs
                    'coordinates': numpy array (:,3) with cartesian coordinates
                    'targets': Dict whose keys are the target_names in the
                        target argument and whose values are numpy arrays
                        with the ANI-1 data

    Returns:
        pd.DataFrame: A dataframe with the columns 'name', 'iconfig',
            'atomic_numbers', and 'coordinates' from the input data. For each target
            in the input data, a column with the target name is added to the
            dataframe with the prefix 'target_'.
    """
    data_unnested_targets = [
        unnest_dictionary(mol, "targets", prefix="target_") for mol in data
    ]
    df = pd.DataFrame().from_records(data_unnested_targets)
    return df


def load_ani1_data(
        config: Dict = ani1_config,
        ani1_path: str = "./ANI-1ccx_clean_fullentry.h5",
        as_dataframe=False,
) -> List[Dict]:
    molecules = get_ani1data(
        allowed_Z=config["allowed_Z"],
        heavy_atoms=config["heavy_atoms"],
        max_config=config["max_config"],
        target=config["target"],
        ani1_path=ani1_path,
    )

    # Some of the molecules have NaN values for some of the targets.
    # This is a problem for the OLS solver, so we drop these molecules.
    molecules = [
        m for m in molecules if not np.isnan(list(m["targets"].values())).any()
    ]

    if as_dataframe:
        return convert_ani1_data_to_dataframe(molecules)

    return molecules
