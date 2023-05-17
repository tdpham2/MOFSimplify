### Modified from app.py from MOFSimplify repository. 
### Objective: Run thermal ANN/ solvent ANN for a large number of CIFs

import subprocess
from pymatgen.core import Structure
from molSimplify.Informatics.MOF.MOF_descriptors import get_primitive, get_MOF_descriptors
import numpy as np
import pandas as pd
import os
import shutil
import keras
from sklearn import preprocessing
import keras.backend as K
from sklearn.metrics import pairwise_distances
import glob
import pymatgen as pm

def descriptor_generator(name, structure, cif_folder, prediction_type, is_entry):
    """
    # descriptor_generator is used by both ss_predict() and ts_predict() to generate RACs and Zeo++ descriptors.
    # These descriptors are subsequently used in ss_predict() and ts_predict() for the ANN models.
    # Inputs are the name of the MOF and the structure (cif file text) of the MOF for which descriptors are to be generated.
    # The third input indicates the type of prediction (solvent removal or thermal).

    :param name: str, the name of the MOF being analyzed.
    :param structure: str, the text of the cif file of the MOF being analyzed.
    :param prediction_type: str, the type of prediction being run. Can either be 'solvent' or 'thermal'.
    :param is_entry: boolean, indicates whether the descriptor CSV has already been written.
    :return: Depends, either the string 'FAILED' if descriptor generation fails, a dictionary myDict (if the MOF being analyzed is in the training data), or an array myResult (if the MOF being analyzed is not in the training data) 
    """ 

    temp_file_folder = MOFSIMPLIFY_PATH + '/' + "temp_file_creation" + '/'

    # Write the data back to a cif file.
    try:
        cif_file = open(cif_folder + name + '.cif', 'r')
    except FileNotFoundError:
        return 'FAILED'
    cif_file.close()
    # There can be a RACs folder for solvent predictions and a RACs folder for thermal predictions. Same for Zeo++.
    RACs_folder = temp_file_folder +  prediction_type + '_RACs/'
    zeo_folder = temp_file_folder + prediction_type + '_zeo++/'

    # Delete the RACs folder, then remake it (to start fresh for this prediction).
    try:
        shutil.rmtree(RACs_folder)
        shutil.rmtree(temp_file_folder + '/' + 'merged_descriptors/')
        shutil.rmtree(zeo_folder)
    except FileNotFoundError:
        pass
    subprocess.run("mkdir -p {}".format(RACs_folder), shell=True)
    subprocess.run("mkdir -p {}/merged_descriptors".format(temp_file_folder), shell=True)
    subprocess.run("mkdir -p {}".format(zeo_folder), shell=True)

    if not is_entry: # have to generate the CSV
        # Rewrite creating primitive cell b/c could not get the get_primitive function to works
        mof = Structure.from_file('{}/{}.cif'.format(cif_folder, name), primitive=True)
        mof.to("{}/{}_primitive.cif".format(cif_folder, name))

        # Next, running MOF featurization
        #try:
        #    get_primitive(cif_folder + name + '.cif', cif_folder + name + '_primitive.cif');
        #except ValueError:
        #    print("FAILED HERE")
        #    return 'FAILED'

        # get_MOF_descriptors is used in RAC_getter.py to get RAC features.
            # The files that are generated from RAC_getter.py: lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv

        # cmd1, cmd2, and cmd3 are for Zeo++. cm4 is for RACs.
        cmd1 = MOFSIMPLIFY_PATH + 'zeo++-0.3/network -ha -res ' + zeo_folder + name + '_pd.txt ' + cif_folder + name + '_primitive.cif'
        cmd2 = MOFSIMPLIFY_PATH + 'zeo++-0.3/network -sa 1.86 1.86 10000 ' + zeo_folder + name + '_sa.txt ' + cif_folder + name + '_primitive.cif'
        cmd3 = MOFSIMPLIFY_PATH + 'zeo++-0.3/network -volpo 1.86 1.86 10000 ' + zeo_folder + name + '_pov.txt '+ cif_folder + name + '_primitive.cif'
        cmd4 = 'python ' + MOFSIMPLIFY_PATH + 'model/RAC_getter.py %s %s %s' %(cif_folder, name, RACs_folder)

        # four parallelized Zeo++ and RAC commands
        process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
        process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
        process3 = subprocess.Popen(cmd3, stdout=subprocess.PIPE, stderr=None, shell=True)
        process4 = subprocess.Popen(cmd4, stdout=subprocess.PIPE, stderr=None, shell=True)

        output1 = process1.communicate()[0]
        output2 = process2.communicate()[0]
        output3 = process3.communicate()[0]
        output4 = process4.communicate()[0]

        # Have written output of Zeo++ commands to files. Now, code below extracts information from those files.

        ''' The geometric descriptors are largest included sphere (Di), 
        largest free sphere (Df), largest included sphere along free path (Dif),
        crystal density (rho), volumetric surface area (VSA), gravimetric surface (GSA), 
        volumetric pore volume (VPOV) and gravimetric pore volume (GPOV). 
        Also, we include cell volume as a descriptor.

        All Zeo++ calculations use a probe radius of 1.86 angstrom, and zeo++ is called by subprocess.
        '''

        dict_list = []
        cif_file = name + '_primitive.cif' 
        basename = cif_file.strip('.cif')
        largest_included_sphere, largest_free_sphere, largest_included_sphere_along_free_sphere_path  = np.nan, np.nan, np.nan
        unit_cell_volume, crystal_density, VSA, GSA  = np.nan, np.nan, np.nan, np.nan
        VPOV, GPOV = np.nan, np.nan
        POAV, PONAV, GPOAV, GPONAV, POAV_volume_fraction, PONAV_volume_fraction = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if (os.path.exists(zeo_folder + name + '_pd.txt') & os.path.exists(zeo_folder + name + '_sa.txt') &
            os.path.exists(zeo_folder + name + '_pov.txt')):
            with open(zeo_folder + name + '_pd.txt') as f:
                pore_diameter_data = f.readlines()
                for row in pore_diameter_data:
                    largest_included_sphere = float(row.split()[1]) # largest included sphere
                    largest_free_sphere = float(row.split()[2]) # largest free sphere
                    largest_included_sphere_along_free_sphere_path = float(row.split()[3]) # largest included sphere along free sphere path
            with open(zeo_folder + name + '_sa.txt') as f:
                surface_area_data = f.readlines()
                for i, row in enumerate(surface_area_data):
                    if i == 0:
                        unit_cell_volume = float(row.split('Unitcell_volume:')[1].split()[0]) # unit cell volume
                        crystal_density = float(row.split('Unitcell_volume:')[1].split()[0]) # crystal density
                        VSA = float(row.split('ASA_m^2/cm^3:')[1].split()[0]) # volumetric surface area
                        GSA = float(row.split('ASA_m^2/g:')[1].split()[0]) # gravimetric surface area
            with open(zeo_folder + name + '_pov.txt') as f:
                pore_volume_data = f.readlines()
                for i, row in enumerate(pore_volume_data):
                    if i == 0:
                        density = float(row.split('Density:')[1].split()[0])
                        POAV = float(row.split('POAV_A^3:')[1].split()[0]) # Probe accessible pore volume
                        PONAV = float(row.split('PONAV_A^3:')[1].split()[0]) # Probe non-accessible probe volume
                        GPOAV = float(row.split('POAV_cm^3/g:')[1].split()[0])
                        GPONAV = float(row.split('PONAV_cm^3/g:')[1].split()[0])
                        POAV_volume_fraction = float(row.split('POAV_Volume_fraction:')[1].split()[0]) # probe accessible volume fraction
                        PONAV_volume_fraction = float(row.split('PONAV_Volume_fraction:')[1].split()[0]) # probe non accessible volume fraction
                        VPOV = POAV_volume_fraction+PONAV_volume_fraction
                        GPOV = VPOV/density
        else:
            print('Not all 3 files exist, so at least one Zeo++ call failed!', 'sa: ',os.path.exists(zeo_folder + name + '_sa.txt'), 
                  '; pd: ',os.path.exists(zeo_folder + name + '_pd.txt'), '; pov: ', os.path.exists(zeo_folder + name + '_pov.txt'))
            return 'FAILED'
        geo_dict = {'name':basename, 'cif_file':cif_file, 'Di':largest_included_sphere, 'Df': largest_free_sphere, 'Dif': largest_included_sphere_along_free_sphere_path,
                    'rho': crystal_density, 'VSA':VSA, 'GSA': GSA, 'VPOV': VPOV, 'GPOV':GPOV, 'POAV_vol_frac':POAV_volume_fraction, 
                    'PONAV_vol_frac':PONAV_volume_fraction, 'GPOAV':GPOAV,'GPONAV':GPONAV,'POAV':POAV,'PONAV':PONAV}
        dict_list.append(geo_dict)
        geo_df = pd.DataFrame(dict_list)
        geo_df.to_csv(zeo_folder + 'geometric_parameters.csv',index=False)

        # error handling for cmd4
        with open(RACs_folder + 'RAC_getter_log.txt', 'r') as f:
            if f.readline() == 'FAILED':
                print('RAC generation failed.')
                return 'FAILED'

        # Merging geometric information with the RAC information that is in the get_MOF_descriptors-generated files (lc_descriptors.csv, sbu_descriptors.csv, linker_descriptors.csv)
        try:
            lc_df = pd.read_csv(RACs_folder + "lc_descriptors.csv") 
            sbu_df = pd.read_csv(RACs_folder + "sbu_descriptors.csv")
            linker_df = pd.read_csv(RACs_folder + "linker_descriptors.csv")
        except Exception: # csv files have been deleted
            return 'FAILED' 

        lc_df = lc_df.mean().to_frame().transpose() # averaging over all rows. Convert resulting Series into a DataFrame, then transpose
        sbu_df = sbu_df.mean().to_frame().transpose()
        linker_df = linker_df.mean().to_frame().transpose()

        merged_df = pd.concat([geo_df, lc_df, sbu_df, linker_df], axis=1)

        merged_df.to_csv(temp_file_folder + '/merged_descriptors/' + name + '_descriptors.csv',index=False) # written in /temp_file_creation_SESSIONID

    else: # CSV is already written
        merged_df = pd.read_csv(temp_file_folder + '/merged_descriptors/' + name + '_descriptors.csv')

    if prediction_type == 'solvent':

        ANN_folder = MOFSIMPLIFY_PATH + 'model/solvent/ANN/'
        train_df = pd.read_csv(ANN_folder + 'dropped_connectivity_dupes/train.csv')

    if prediction_type == 'thermal':
        ANN_folder = MOFSIMPLIFY_PATH + 'model/thermal/ANN/'
        train_df = pd.read_csv(ANN_folder + 'train.csv')

    myResult = [temp_file_folder, ANN_folder]

    return myResult 

def normalize_data_thermal(df_train, df_newMOF, fnames, lname, debug=False): # Function assumes it gets pandas DataFrames with MOFs as rows and features as columns
    """
    normalize_data_thermal takes in two DataFrames df_train and df_newMOF, one for the training data (many rows) and one for the new MOF (one row) for which a prediction is to be generated.
    This function also takes in fnames (the feature names) and lname (the target property name).
    This function normalizes the X values from the pandas DataFrames and returns them as X_train and X_newMOF.
    It also normalizes y_train, which are the thermal breakdown temperatures in the training data DataFrame, and returns x_scaler (which scaled X_train) and y_scaler (which scaled y_train).

    :param df_train: A pandas DataFrame of the training data.
    :param df_newMOF: A pandas DataFrame of the new MOF being analyzed.
    :param fnames: An array of column names of the descriptors.
    :param lname: An array of the column name of the target.
    :param debug: A boolean that determines whether extra information is printed.
    :return: numpy.ndarray X_train, the descriptors of the training data. Its number of rows is the number of MOFs in the training data. Its number of columns is the number of descriptors.
    :return: numpy.ndarray X_newMOF, the descriptors of the new MOF being analyzed by MOFSimplify. It contains only one row.
    :return: numpy.ndarray y_train, the thermal stabilities of the training data. 
    :return: sklearn.preprocessing._data.StandardScaler x_scaler, the scaler used to normalize the descriptor data to unit mean and a variance of 1. 
    :return: sklearn.preprocessing._data.StandardScaler y_scaler, the scaler used to normalize the target data to unit mean and a variance of 1.
    """
    _df_train = df_train.copy().dropna(subset=fnames+lname)
    _df_newMOF = df_newMOF.copy().dropna(subset=fnames)
    X_train, X_newMOF = _df_train[fnames].values, _df_newMOF[fnames].values # takes care of ensuring ordering is same for both X
    y_train = _df_train[lname].values
    if debug:
        print("training data reduced from %d -> %d because of nan." % (len(df_train), y_train.shape[0]))
    #x_scaler = sklearn.preprocessing.StandardScaler()
    x_scaler = preprocessing.StandardScaler()
    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_newMOF = x_scaler.transform(X_newMOF)
    #y_scaler = sklearn.preprocessing.StandardScaler()
    y_scaler = preprocessing.StandardScaler()
    y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    return X_train, X_newMOF, y_train, x_scaler, y_scaler

def run_thermal_ANN(path, MOF_name, thermal_ANN):
    """
    run_thermal_ANN runs the thermal stability ANN with the desired MOF as input.
    It returns a prediction for the thermal breakdown temperature of the chosen MOF.

    :param user_id: str, the session ID of the user
    :param path: str, the server's path to the MOFSimplify folder on the server
    :param MOF_name: str, the name of the MOF for which a prediction is being generated
    :param thermal_ANN: keras.engine.training.Model, the ANN itself
    :return: str new_MOF_pred, the model thermal stability prediction 
    :return: list neighbor_names, the latent space nearest neighbor MOFs in the thermal stability ANN
    :return: list neighbor_distances, the latent space distances of the latent space nearest neighbor MOFs in neighbor_names 
    """ 

    RACs = ['D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all',
     'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all', 'D_func-S-3-all',
     'D_func-T-0-all', 'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all',
     'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all', 'D_func-Z-3-all',
     'D_func-chi-0-all', 'D_func-chi-1-all', 'D_func-chi-2-all',
     'D_func-chi-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all', 'D_lc-I-2-all',
     'D_lc-I-3-all', 'D_lc-S-0-all', 'D_lc-S-1-all', 'D_lc-S-2-all',
     'D_lc-S-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all',
     'D_lc-T-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all',
     'D_lc-Z-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
     'D_lc-chi-3-all', 'D_mc-I-0-all', 'D_mc-I-1-all', 'D_mc-I-2-all',
     'D_mc-I-3-all', 'D_mc-S-0-all', 'D_mc-S-1-all', 'D_mc-S-2-all',
     'D_mc-S-3-all', 'D_mc-T-0-all', 'D_mc-T-1-all', 'D_mc-T-2-all',
     'D_mc-T-3-all', 'D_mc-Z-0-all', 'D_mc-Z-1-all', 'D_mc-Z-2-all',
     'D_mc-Z-3-all', 'D_mc-chi-0-all', 'D_mc-chi-1-all', 'D_mc-chi-2-all',
     'D_mc-chi-3-all', 'f-I-0-all', 'f-I-1-all', 'f-I-2-all', 'f-I-3-all',
     'f-S-0-all', 'f-S-1-all', 'f-S-2-all', 'f-S-3-all', 'f-T-0-all', 'f-T-1-all',
     'f-T-2-all', 'f-T-3-all', 'f-Z-0-all', 'f-Z-1-all', 'f-Z-2-all', 'f-Z-3-all',
     'f-chi-0-all', 'f-chi-1-all', 'f-chi-2-all', 'f-chi-3-all', 'f-lig-I-0',
     'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-S-0', 'f-lig-S-1', 'f-lig-S-2',
     'f-lig-S-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-Z-0',
     'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3', 'f-lig-chi-0', 'f-lig-chi-1',
     'f-lig-chi-2', 'f-lig-chi-3', 'func-I-0-all', 'func-I-1-all',
     'func-I-2-all', 'func-I-3-all', 'func-S-0-all', 'func-S-1-all',
     'func-S-2-all', 'func-S-3-all', 'func-T-0-all', 'func-T-1-all',
     'func-T-2-all', 'func-T-3-all', 'func-Z-0-all', 'func-Z-1-all',
     'func-Z-2-all', 'func-Z-3-all', 'func-chi-0-all', 'func-chi-1-all',
     'func-chi-2-all', 'func-chi-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all',
     'lc-I-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all',
     'lc-T-0-all', 'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-Z-0-all',
     'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-chi-0-all', 'lc-chi-1-all',
     'lc-chi-2-all', 'lc-chi-3-all', 'mc-I-0-all', 'mc-I-1-all', 'mc-I-2-all',
     'mc-I-3-all', 'mc-S-0-all', 'mc-S-1-all', 'mc-S-2-all', 'mc-S-3-all',
     'mc-T-0-all', 'mc-T-1-all', 'mc-T-2-all', 'mc-T-3-all', 'mc-Z-0-all',
     'mc-Z-1-all', 'mc-Z-2-all', 'mc-Z-3-all', 'mc-chi-0-all', 'mc-chi-1-all',
     'mc-chi-2-all', 'mc-chi-3-all']
    geo = ['Df','Di', 'Dif','GPOAV','GPONAV','GPOV','GSA','POAV','POAV_vol_frac',
      'PONAV','PONAV_vol_frac','VPOV','VSA','rho']
     
    other = ['cif_file','name','filename']

    ANN_path = path + 'model/thermal/ANN/'
    temp_file_path = path + 'temp_file_creation' +  '/'
    df_train_all = pd.read_csv(ANN_path+"train.csv").append(pd.read_csv(ANN_path+"val.csv"))
    df_train = pd.read_csv(ANN_path+"train.csv")
    df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]
    df_newMOF = pd.read_csv(temp_file_path + 'merged_descriptors/' + MOF_name + '_descriptors.csv') # Assume temp_file_creation/ in parent directory
    features = [val for val in df_train.columns.values if val in RACs+geo]

    X_train, X_newMOF, y_train, x_scaler, y_scaler = normalize_data_thermal(df_train, df_newMOF, features, ["T"], debug=False)
    X_train.shape, y_train.reshape(-1, ).shape 

    model = thermal_ANN

    #from tensorflow.python.keras.backend import set_session
    #with tf_session.as_default():
    #    with tf_session.graph.as_default():
    new_MOF_pred = y_scaler.inverse_transform(model.predict(X_newMOF))
    new_MOF_pred = np.round(new_MOF_pred,1) # round to 1 decimal

    # isolating just the prediction, since the model spits out the prediction like [[PREDICTION]], as in, in hard brackets
    new_MOF_pred = new_MOF_pred[0][0]
    new_MOF_pred = str(new_MOF_pred)

    # adding units
    degree_sign= u'\N{DEGREE SIGN}'
    new_MOF_pred = new_MOF_pred + degree_sign + 'C' # degrees Celsius

    # Define the function for the latent space. This will depend on the model. We want the layer before the last, in this case this was the 8th one.
    get_latent = K.function([model.layers[0].input],
                            [model.layers[8].output]) # Last layer before dense-last
    # Get the latent vectors for the training data first, then the latent vectors for the test data.

    # TODO: get_latent function format is different
    #training_latent = get_latent([X_train, 0])[0]
    training_latent = get_latent(X_train)[0]
    #design_latent = get_latent([X_newMOF, 0])[0]
    design_latent = get_latent(X_newMOF)[0]

    # Compute the pairwise distances between the test latent vectors and the train latent vectors to get latent distances
    d1 = pairwise_distances(design_latent,training_latent,n_jobs=30)
    df1 = pd.DataFrame(data=d1, columns=df_train['CoRE_name'].tolist())
    df1.to_csv(temp_file_path + 'solvent_test_latent_dists.csv')

    # Want to find the closest points (let's say the closest 5 points); so, smallest values in df1
    neighbors = 5 # number of closest points

    # will make arrays of length neighbors, where each entry is the next closest neighbor (will do this for both names and distances)
    neighbors_names = []
    neighbors_distances = []

    df_reformat = df1.min(axis='index')

    for i in range(neighbors):
        name = df_reformat.idxmin() # name of next closest complex in the training data
        distance = df_reformat.min() # distance of the next closest complex in the training data to the new MOF
        df_reformat = df_reformat.drop(name) # dropping the next closest complex, in order to find the next-next closest complex

        neighbors_names.append(name)
        neighbors_distances.append(str(distance))

    return new_MOF_pred, neighbors_names, neighbors_distances

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

### Main part ###
MOFSIMPLIFY_PATH = os.path.abspath('.') + '/'
thermal_ANN_path = MOFSIMPLIFY_PATH + 'model/thermal/ANN/'
dependencies = {'precision':precision,'recall':recall,'f1':f1}
thermal_model = keras.models.load_model(thermal_ANN_path + 'final_model_T_few_epochs.h5',custom_objects=dependencies)
cif_folder = './cifs/'

cifs = sorted(glob.glob(cif_folder + '/*.cif'))
# Ignore CIF with primitive in name, in case of duplication
for i in cifs:
    if 'primitive' in i:
        cifs.remove(i)

n = [i.split('/')[-1] for i in cifs]
names = [i.split('.')[0] for i in n]

for cif, name in zip(cifs, names):
    f = open(cif, 'r')
    data = f.readlines()
    structure=''.join(data)
    f.close()
    
    # Generate descriptor
    code = descriptor_generator(name, structure, cif_folder, 'thermal', False)
    if code == 'FAILED':
        with open("failed_MOFs_1.csv", "a") as f:
            f.write("{}\n".format(name))
        continue
    # Make prediction
    prediction, neighbor_names, neighbor_distances = run_thermal_ANN(MOFSIMPLIFY_PATH, name, thermal_model)
    print(name, prediction)
    with open('Thermal_ANN.csv', 'a') as f:
        f.write("{},{}\n".format(name, prediction))
