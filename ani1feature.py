"""Usage: from ani1_feature import ani1_feature
To run a unit test: python ani_feature.py"""
import numpy as np


def ani1_feature(Z, R):
    """Returns np.array shape=(num_atoms, 384)
        ani1 feature vector.
    Args:
        Z: np.array shape=(num_atoms,), atom numbers
        R: np.array shape=(3, num_atoms), xyz coordinates of atoms
    """
    assert len(Z) == R.shape[1]
    assert R.shape[0] == 3
    atom_Z_arr = np.asarray(Z)
    atom_R_arr = R.T
    '''constants'''
    rad_cutoff = 4.6
    ang_cutoff = 3.1
    rad_shifts = np.asarray([
        5.0000000e-01, 7.5625000e-01, 1.0125000e+00, 1.2687500e+00,
        1.5250000e+00, 1.7812500e+00, 2.0375000e+00, 2.2937500e+00,
        2.5500000e+00, 2.8062500e+00, 3.0625000e+00, 3.3187500e+00,
        3.5750000e+00, 3.8312500e+00, 4.0875000e+00, 4.3437500e+00])
    rad_eta = 1.6000000e+01
    zeta = 8.0000000e+00
    ang_theta_shifts = np.asarray([
        0.0000000e+00, 7.8539816e-01, 1.5707963e+00, 2.3561945e+00,
        3.1415927e+00, 3.9269908e+00, 4.7123890e+00, 5.4977871e+00])
    ang_eta = 6.0000000e+00
    ang_shifts = np.asarray([
        5.0000000e-01, 1.1500000e+00, 1.8000000e+00, 2.4500000e+00])


    '''compute'''
    ref_Z_list = [1, 6, 7, 8]
    #ref_Z_list = [1,6,8]
    ref_Z_indices_list = [atom_Z_arr == ref_Z for ref_Z in ref_Z_list]
    atom_feat_list = []
    for atom_i, (atom_Z, atom_R) in enumerate(zip(atom_Z_arr, atom_R_arr)):
        atom_feat = []
        for ref_Z_i, ref_Z_indices1 in enumerate(ref_Z_indices_list):
            env_indices1 = ref_Z_indices1.copy()
            env_indices1[atom_i] = False
            env_R_arr1 = atom_R_arr[env_indices1]
            dist1 = calc_dist(atom_R, env_R_arr1)
            rad_func = piece_wise_cutoff(dist1, rad_cutoff)
            rad_exp = np.exp(-rad_eta * (dist1 - rad_shifts[:, np.newaxis])**2)  # num_shifts by num_env
            rad_vec = rad_exp.dot(rad_func)
            atom_feat.append(rad_vec)
            for ref_Z_indices2 in ref_Z_indices_list[ref_Z_i:]:
                env_indices2 = ref_Z_indices2.copy()
                env_indices2[atom_i] = False
                env_R_arr2 = atom_R_arr[env_indices2]
                dist2 = calc_dist(atom_R, env_R_arr2)
                ang_func1 = piece_wise_cutoff(dist1, ang_cutoff)
                ang_func2 = piece_wise_cutoff(dist2, ang_cutoff)
                prod_func12 = ang_func1[:, np.newaxis] * ang_func2
                angle = calc_angle(atom_R, env_R_arr1, env_R_arr2)

                cos_factor = (1.0 + np.cos(angle[np.newaxis, :, :] - ang_theta_shifts[:, np.newaxis, np.newaxis]))**zeta  # num_theta_shifts by num_env1 by num_env2
                ang_exp = np.exp(-ang_eta * (((dist1[:, np.newaxis] + dist2) / 2)[np.newaxis, :, :] - ang_shifts[:, np.newaxis, np.newaxis])**2)  # num_shifts by num_env1 by num_env2
                cos_exp = cos_factor[:, np.newaxis, :, :] * ang_exp[np.newaxis, :, :, :]

                ang_vec = (cos_exp * prod_func12[np.newaxis, np.newaxis, :, :]).sum(axis=-1).sum(axis=-1).ravel()
                ang_vec *= 2**(1.0 - zeta)
                atom_feat.append(ang_vec)
        atom_feat_list.append(np.concatenate(atom_feat))
    return np.stack(atom_feat_list)




def calc_dist(atom_R, env_R_arr):
    """Returns np.array shape=(num_atoms,)
        vectorized distances between an atom and an atomic env.
    Args:
        atom_R: np.array shape=(3,), xyz coordinates of an atom
        env_R_arr: np.array shape=(num_atoms, 3), xyz coordinates of env atoms
    Note:
        env_R_arr must not contain atom_R
    """
    #assert atom_R not in env_R_arr
    res = np.linalg.norm(atom_R - env_R_arr, axis=1)
    if len(res) > 0:
        assert np.min(res) > 1.0e-5
    return res

def calc_angle(atom_R, env_R_arr1, env_R_arr2):
    """Returns np.array shape=(num_atoms1, num_atom2)
        vectorized angles between an atom and two atomic envs.
    Args:
        atom_R: np.array shape=(3,), xyz coordinates of an atom
        env_R_arr1: np.array shape=(num_atoms, 3), xyz of env1 atoms
        env_R_arr2: np.array shape=(num_atoms, 3), xyz of env2 atoms
    Note:
        env_R_arr1 or env_R_arr2 must not contain atom_R
    """
    #assert atom_R not in env_R_arr1
    #assert atom_R not in env_R_arr2
    atom_env1 = env_R_arr1 - atom_R  # (num_env1, 3)
    atom_env2 = env_R_arr2 - atom_R  # (num_env2, 3))
    norm1 = np.linalg.norm(atom_env1, axis=1)
    if len(norm1) > 0:
        assert np.min(norm1) > 1.0e-5
    norm2 = np.linalg.norm(atom_env2, axis=1)
    if len(norm2) > 0:
        assert np.min(norm2) > 1.0e-5
    cosine = atom_env1.dot(atom_env2.T) / np.outer(norm1, norm2)
    cosine[cosine > 1.0] = 1.0
    return np.arccos(cosine)



def piece_wise_cutoff(dist, cutoff):
    """Returns np.array shape=(num_atoms,)
        vectorized piece-wise cutoff functions of an atom in an atomic env
    Args:
        dist: np.array shape=(num_atoms,), atom-env distances
    """
    return (0.5 * np.cos(np.pi * dist / cutoff) + 0.5) * (dist <= cutoff)

if __name__ == '__main__':
    test_Z = np.asarray([1, 1, 1, 1, 1, 1, 6, 6])
    test_R = np.asarray([[ 1.14708126,  0.92832923,  1.23854411, -0.97821295, -1.1680795 ,
        -1.16438651,  0.75624859, -0.75682282],
       [ 1.00382483, -0.44134787, -0.55805689,  0.86605525, -0.87597656,
         0.00876343, -0.00498759,  0.0082047 ],
       [-0.3857688 ,  1.10483515, -0.42403421,  0.84534216,  0.09407744,
        -0.91788441, -0.03900506,  0.01159992]])
    test_feat = ani1_feature(test_Z, test_R)
    print((test_feat.max(), test_feat.min()))
    print((test_feat.shape))

    test_Z = np.asarray([1, 1, 8])
    test_R = np.asarray([[ 0.        ,  0.        ,  0.        ],
       [ 0.74262547, -0.84631765,  0.00653355],
       [-0.32607275, -0.40695924,  0.10498759]])
    test_feat2 = ani1_feature(test_Z, test_R)
    print((test_feat2.max(), test_feat2.min()))
    print((test_feat2.shape))

