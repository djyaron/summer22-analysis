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
#%% Imports, definitions

import itertools
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from ani1_interface import get_ani1data
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

Array = np.ndarray


#%% Code behind


def build_XX_matrix(dataset: List[Dict], allowed_Zs: List[int]) -> Array:
    nmol = len(dataset)
    XX = np.zeros([nmol, len(allowed_Zs) + 1])
    iZ = {x: i for i, x in enumerate(allowed_Zs)}

    for imol, molecule in enumerate(dataset):
        Zc = Counter(molecule["atomic_numbers"])
        for Z, count in Zc.items():
            # TODO: The XX matrix is uniquely determined by `dataset` and
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


def create_heatmap(
    molecules: List[Dict],
    target: str,
    allowed_Z: List[int],
    plot_args: Optional[Dict] = None,
    show_progress: bool = False,
    XX: Optional[Array] = None,
) -> plt.Axes:
    """Creates a heatmap of the MAE between methods.

    Args:
        molecules (List[Dict]): From ANI-1 dataset
        target (str): List of method IDs to compare
        allowed_Z (List[int]): The allowed atoms in the molecules
        plot_args (Optional[Dict]): Arguments to pass to seaborn heatmap
        show_progress (bool): Show TQDM progress bar

    Returns:
        plt.Axes: Matplotlib axes object
    """
    n_targets = len(target.keys())
    mae_matrix = np.zeros((n_targets, n_targets))

    target_keys = list(target.keys())
    target_values = list(target.values())

    # List indices
    target_idx_pairs = list(itertools.combinations(range(n_targets), 2))

    conversion = 627.50961

    if show_progress:
        target_idx_pairs = tqdm(target_idx_pairs)

    for (idx_1, idx_2) in target_idx_pairs:
        coefs, XX, method1_mat, method2_mat = fit_linear_ref_ener2(
            molecules, target_keys[idx_1], target_keys[idx_2], allowed_Z, XX=XX
        )

        resid = method2_mat - (method1_mat + (XX @ coefs))
        mae_matrix[idx_2, idx_1] = np.mean(np.abs(resid))

    mae_matrix = mae_matrix * conversion

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


#%% Main block

# https://drive.google.com/file/d/1SP8SX0v5d1UJAX69GpMV-JtjfUSnf-QB
ani1_path = "../../Data/ANI-1ccx_clean_fullentry.h5"
molecules_path = "../../Data/ani1-extracted.pkl"

ani1_config = {
    "allowed_Z": [1, 6, 7, 8],
    "heavy_atoms": list(range(1, 9)),
    "max_config": 1_000_000,
    "target": {
        "dt": "dftb.energy",  # Dftb Total
        # "de": "dftb.elec_energy",  # Dftb Electronic
        # "dr": "dftb.rep_energy",  # Dftb Repulsive
        "pt": "dftb_plus.energy",  # dftb Plus Total
        # "pe": "dftb_plus.elec_energy",  # dftb Plus Electronic
        # "pr": "dftb_plus.rep_energy",  # dftb Plus Repulsive
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

XX = build_XX_matrix(molecules, ani1_config["allowed_Z"])

#%%

# Create a heatmap of the MAE between methods
fig, ax = plt.subplots(figsize=(15, 15))
create_heatmap(
    molecules,
    ani1_config["target"],
    ani1_config["allowed_Z"],
    show_progress=True,
)

plt.show()

# %%
