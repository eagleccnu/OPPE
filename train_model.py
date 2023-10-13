"""
this script is used to train the model for predicting protein order parameter
you can train your own model by changing the model parameters
the model is based on random forest (RF)
"""

import numpy as np
import dill
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os
import joblib
import utils


csv_file = "pdb_bmrb_pair.csv"
dir_pdb = "pdb_train/"
dir_nmrstar = "nmrstar/"

dict_pdb_bmrb = utils.read_pdb_bmrb_dict_from_csv(csv_file)
list_pdb_file = os.listdir(dir_pdb)


def add_flag(array):
    array = array.tolist()
    new_array = []
    for i in range(len(array)):
        if i < 3 or i > len(array)-4:
            new_array.append([1]+array[i])
        else:
            new_array.append([0]+array[i])
    return np.array(new_array)


# data preprocessing to obtain feature and label
def data_preprocessing():

    for i, pdb_file in enumerate(list_pdb_file):

        pdb_file_full_path = dir_pdb + pdb_file
        pdb_id = utils.read_pdbid_from_filename(pdb_file)

        bmrb_id = dict_pdb_bmrb[pdb_id.upper()]
        bmrb_file = utils.search_file_with_bmrb(bmrb_id, dir_nmrstar)
        bmrb_file_full_path = dir_nmrstar + bmrb_file
        # print("{} - {}".format(pdb_id, bmrb_id))

        protein_temp = utils.protein_s2(pdb_id, bmrb_id)
        protein_temp.read_seq_train(pdb_file_full_path, bmrb_file_full_path)

        protein_temp.read_dist_var_from_pdb(pdb_id, pdb_file_full_path)
        protein_temp.read_torsion_var_from_pdb(pdb_id, pdb_file_full_path)
        protein_temp.read_s2_from_star(bmrb_file_full_path)
        protein_temp.cal_ss_from_pdb(pdb_id, pdb_file_full_path)
        protein_temp.get_rsa(pdb_id, pdb_file_full_path)
        protein_temp.read_concat_num_from_pdb(pdb_id,pdb_file_full_path)

        protein_temp.merge_seq()
        array_data_temp = protein_temp.to_numpy()

        array_data_temp = add_flag(array_data_temp)

        if i == 0:
            array_data = array_data_temp
        else:
            array_data = np.concatenate((array_data, array_data_temp), axis=0)

    print("saving data...")
    with open('numpy_dataset_training.dat', 'wb') as f:
        dill.dump(array_data, f)

    # random shuffle numpy array
    np.random.seed(100)
    np.random.shuffle(array_data)

    scaler = RobustScaler()
    array_data[:, 3: -1] = scaler.fit_transform(array_data[:, 3: -1])  # leave the last column, s2, unchange

    feature = array_data[:, : -1]
    label = array_data[:, -1]

    return feature, label


#####################################################
####   train RF model
#####################################################

forest = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1)
feature, label = data_preprocessing()
forest.fit(feature, label)
scores = cross_val_score(forest, feature, label, cv=5, scoring='neg_mean_absolute_error')  # 5-fold cross-validation
# print('neg_mean_absolute_error: ', scores.mean())

# save model
joblib.dump(forest, 'model.pkl')


#####################################################
####   sort feature importance
#####################################################

feature_list_sort = ['CN', 'SASA', 'SS_H', 'SS_E', 'SS_C', 'Flag', 'PHI_var', 'PSI_var',
                     'D_var(i,-3)', 'D_var(i,-2)', 'D_var(i,-1)', 'D_var(i,1)', 'D_var(i,2)', 'D_var(i,3)'
                     ]

feature_importance = forest.feature_importances_.tolist()
feature_importance2 = feature_importance[12:14]+feature_importance[1:4]+[feature_importance[0]]+feature_importance[4:12]
x_values = list(range(len(feature_importance)))

# plot feature importance
plt.figure(figsize=(18, 8))
plt.xticks(size=18)
plt.yticks(x_values, feature_list_sort, rotation='horizontal', size=15)
plt.barh(x_values, feature_importance2)
plt.xlabel('Importance', fontsize=18)
plt.ylabel('Feature', fontsize=18)
# plt.savefig('FeatureImportance1.jpg')

# six features importance sort
feature_value = [sum(feature_importance[6:12]), sum(feature_importance[4:6]), feature_importance[0],
                 sum(feature_importance[1:4]), feature_importance[13], feature_importance[12]]
feature_name = ['Distance', 'Angle', 'Flag', 'Secondary', 'Area', 'Contact']
x_values = list(range(len(feature_name)))
plt.figure()
feature_list_sort = ['']*14
plt.ylabel('Importance', fontsize=12)
plt.yticks(size=10)
plt.xticks(x_values, feature_name, rotation='horizontal', size=12)
plt.bar(x_values, feature_value)
# plt.savefig('FeatureImportance2.tif')
plt.show()


