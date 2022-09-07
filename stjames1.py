#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:32:20 2022

@author: yaron
"""

import os
import pickle
from typing import Any, Dict, List, Tuple, Union
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

from ani1_interface import get_ani1data
from ani1feature import ani1_feature

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

def load_ani1_data(
    config: Dict = ani1_config,
    ani1_path: str = "./data/ANI-1ccx_clean_fullentry.h5",
    as_dataframe=False,
) -> List[Dict]:
    r"""Loads molecules from the ANI-1 Dataset

        Arguments:
            config (Dict): data to grab from ANI-1
            ani1_path (str): ANI-1 dataset
            as_dataframe (bool): return as dataframe or not

        Returns:
            molecules (List[Dict]): molecules from ANI-1 dataset
        """
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

#%%
mols = load_ani1_data()
#%%
test1 = mols[10000]

Z1 = test1['atomic_numbers']
R1 = test1['coordinates'].T


f1 = ani1_feature(Z1, R1)

test2 = mols[10001]

Z2 = test2['atomic_numbers']
R2 = test2['coordinates'].T


f2 = ani1_feature(Z2, R2)

C1 = f1[0,:]
C2 = f2[0,:]

plt.figure(1)
plt.plot(C1,'r.')
plt.figure(2)
plt.plot(C2,'b.')
plt.figure(3)
plt.plot(C1-C2,'g.')

C1n = C1/np.linalg.norm(C1)
C2n = C2/np.linalg.norm(C2)
diag1 = np.dot(C1n.T, C1n)
diag2 = np.dot(C2n.T, C2n)
diff = np.dot(C1n.T, C2n)

#%%
dft = np.array([x['targets']['hd'] for x in mols]) * 627.0 # hartree to kcal/mol
cc = np.array([x['targets']['cc'] for x in mols]) * 627.0
plt.figure(10)
plt.plot(dft-cc, 'r.')
