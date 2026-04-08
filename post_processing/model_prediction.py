import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import h5py
import tensorflow as tf
import numpy as np
import math
import argparse
import scipy.io
import os
from tensorflow import keras
from model_architecture.runet import create_vae

current_job_id = os.environ.get("SLURM_JOB_ID", "no_jobid")

#%% 
parser = argparse.ArgumentParser()
parser.add_argument('-p','--parent_folder')
parser.add_argument('-w','--weight_file', default = 'model_best.weights.h5')
parser.add_argument('-i','--input_file', default = 'input128x128_pred_cases.h5')
parser.add_argument('-o','--output_file', default = 'output128x128_pred_cases.h5')
parser.add_argument('-d','--dataset') #Supply folder name up to the seed
args = parser.parse_args()

#%%
#########################
# Set up project folder
#########################
results_folder = Path(args.parent_folder)
plotting_folder = results_folder / 'plotting_info'
plotting_folder.mkdir(exist_ok=True, parents=True)
weights_file = results_folder / args.weight_file

#%%
dataset_folder = Path(args.dataset)
input_file= dataset_folder / args.input_file
output_file = dataset_folder / args.output_file

with h5py.File(input_file , 'r') as f: 
    X_predict = f['input_data'][:]

with h5py.File(output_file, 'r') as f:
    Y_predict = f['output_data'][:]

print( "=" * 40)
print( "\n")
print("Prediction input shape:",  X_predict.shape)
print("Prediction output shape:",  Y_predict.shape)
print( "\n")

# Read from cmd_arguements of what reg_weight was used in training
with open ( results_folder / 'cmd_arguments.txt', 'r') as f:
    for line in f:
        if line.startswith("Regularization weight:"):
            reg_weight_str = line.strip().split(":")[1].strip()
            reg_weight = float(reg_weight_str)

#%%
#########################
#Load Model Back
#########################
input_size = (X_predict.shape[1], X_predict.shape[2], X_predict.shape[3])
vae_model = create_vae(reg_weight=reg_weight, input_size=input_size, output_channels=Y_predict.shape[3])
vae_model.load_weights(weights_file)
results = vae_model.predict(X_predict, verbose = 2)

# Calculating domain average satruation here
true_domain_sat = tf.reduce_mean(Y_predict, axis=[1,2,3])
pred_domain_sat = tf.reduce_mean(results, axis=[1,2,3])

# Save the results to a .npy file
np.save(plotting_folder / 'predictions.npy', results) 
np.save(plotting_folder / 'y_predict_true.npy', Y_predict)
np.save(plotting_folder / 'true_avg_domain_sat.npy', true_domain_sat)
np.save(plotting_folder / 'pred_avg_domain_sat.npy', pred_domain_sat)

# Append current job id to cmd_arguments.txt file to keep track
with open( results_folder / 'cmd_arguments.txt', 'a') as f:
    f.write(f'Predction job ID: {current_job_id}' + '\n')