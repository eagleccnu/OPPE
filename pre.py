import argparse
import utils
import numpy as np
import dill
import torch
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import joblib


device = torch.device("cpu")
parser = argparse.ArgumentParser(description='Predict Order Parameter based on NMR Structure Ensemble')
parser.add_argument('pdb_file_name', metavar='File', help='file name of PDB')


def main():

    args = parser.parse_args()
    pdb_file_name = args.pdb_file_name

    numpy_file = 'numpy_dataset_training.dat'
    pdb_file_path = 'pdb_test/'
    model_file = 'model.pkl'

    #####################################################
    ####   data preprocessing
    #####################################################
    with open(numpy_file, 'rb') as f:
        numpy_dataset = dill.load(f)
    scaler = RobustScaler()
    scaler.fit(numpy_dataset[:, 3:-1])  # leave the last column, s2, unchange

    dict_2nd_structure_code = {'H': np.array([1, 0, 0]).tolist(), 'E': np.array([0, 1, 0]).tolist(),
                               'C': np.array([0, 0, 1]).tolist()}

    pdb_file = pdb_file_path + pdb_file_name
    pdb_id = pdb_file_name.split('.')[0]

    protein_temp = utils.protein_s2_pre(pdb_id)
    protein_temp.read_seq(pdb_file)

    protein_temp.read_dist_var_from_pdb(pdb_id, pdb_file)
    protein_temp.read_torsion_var_from_pdb(pdb_id, pdb_file)
    protein_temp.read_concat_num_from_pdb(pdb_id, pdb_file)
    protein_temp.cal_ss_from_pdb(pdb_id, pdb_file)
    protein_temp.get_rsa(pdb_id, pdb_file)

    #####################################################
    ####   load trained model, start predict order parameter for every residue
    #####################################################
    model = joblib.load(model_file)

    print('Predict Order Parameter ...')
    s2_pred = []
    with torch.no_grad():
        for i, res in enumerate(protein_temp.pdb_seq):
            if i < 3 or i > protein_temp.length_eff - 4:
                flag = 1
            else:
                flag = 0
            feature = [flag] + dict_2nd_structure_code[res.state_3] + \
                      [res.phi_var, res.psi_var, res.dist_var[0], res.dist_var[1], res.dist_var[2],
                       res.dist_var[3],
                       res.dist_var[4], res.dist_var[5], res.concat_num, res.acc]

            feature = np.array(feature)
            feature = feature[np.newaxis, :]
            feature[:, 3:] = scaler.transform(feature[:, 3:])

            batch_feature = torch.tensor(feature, dtype=torch.float32)
            batch_feature = batch_feature.to(device)

            pred = model.predict(batch_feature)
            pred = pred[0]
            s2_pred.append(pred)
            # print("real S2: {} \t predicted S2: {}".format(res.s2, pred))

        # use the average of three consecutive predictions to represent the middle prediction
        s2_pred_smooth = [0] * len(s2_pred)
        for i in range(1, len(s2_pred) - 1):
            s2_pred_smooth[i] = (s2_pred[i - 1] + s2_pred[i] + s2_pred[i + 1]) / 3
        s2_pred_smooth[0] = s2_pred[0]
        s2_pred_smooth[-1] = s2_pred[-1]
    s2_pred = s2_pred_smooth
    print("predicted S2: {}".format([round(i, 3) for i in s2_pred]))

    # plot result figure
    # plt.figure()
    # plt.ylabel('Prediction of Order Parameter', fontsize=15)
    # plt.xlabel('Index of Residue', fontsize=15)
    # plt.title("PDB ID: {}".format(pdb_id.upper()), size=15)
    # plt.yticks(size=15)
    # plt.xticks(size=15)
    # index_res = range(1, 1 + len(s2_pred))
    # plt.plot(index_res, s2_pred, 'go')
    # plt.ylim((0, 1.0))
    # # plt.savefig(pdb_id + '.jpg')
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


