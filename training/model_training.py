##############################################
# Import libraries
##############################################
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import tensorflow as tf
import numpy as np
import random
import math
import argparse
from scipy.io import savemat
from scipy.io import loadmat
import time
import os
from model_architecture.runet import create_vae
from learning_rate_warmup import LinearWarmupToPiecewiseConstant
from tensorflow import keras
import csv
import gc
import tensorflow.keras.backend as K

##############################################
# Obtain variables needed later 
##############################################
job_id = os.environ.get("SLURM_JOB_ID", "no_jobid")

strategy = tf.distribute.MirroredStrategy()
num_GPUs = strategy.num_replicas_in_sync

#%% Check GPU availability
print( "=" * 40)
print( "\n")
print(f"Num GPUs: {num_GPUs}")
print( "\n")

#%% Obtain input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p','--parent_folder')
parser.add_argument('-ds','--dataset_seed')
parser.add_argument('-b','--base_batch_size', type=int)
parser.add_argument('-e','--epoch', type=int)
parser.add_argument('-el','--epoch_loop',type=int)
parser.add_argument('-l', '--learning_rate', type=float)
parser.add_argument('-tl', '--target_learning_rate', type=float)
parser.add_argument('-dw', '--lr_decay_weight', type=float)
parser.add_argument('-w', '--warm_up_step_percentage', type=float)
parser.add_argument('-r', '--reg_weight',type=float)
parser.add_argument('-tfs','--tf_seed', type=int)
parser.add_argument('-ss','--seed_for_val_shuffling', type=int)
args = parser.parse_args()

base_batch_size = args.base_batch_size
epochs = args.epoch
epoch_loop_num = args.epoch_loop
initial_lr = args.learning_rate
target_lr = args.target_learning_rate
reg_weight = args.reg_weight
global_batch_size = base_batch_size * num_GPUs
warm_up_step_percentage = args.warm_up_step_percentage
lr_decay_weight = args.lr_decay_weight
seed_for_tf = args.tf_seed
seed_for_shuffling = args.seed_for_val_shuffling 

##############################################
# Set seed_for_rng for reproducible results
##############################################
tf.random.set_seed(seed_for_tf)
tf.config.experimental.enable_op_determinism() 

##############################################
# Set up project folder
##############################################
project_path = Path(args.parent_folder)

# This is where you want to save the history and model weights
results_folder = project_path / f'b{base_batch_size}_e{epochs * epoch_loop_num}_lr{initial_lr}_r{reg_weight}'/ f'job_{job_id}'
results_folder.mkdir(parents=True, exist_ok=True)

# Get the training and validation data
dataset_path = project_path / 'training_validation_prediction_data'/ f'seed_{args.dataset_seed}'


with h5py.File(dataset_path / 'input128x128_train.h5'  , 'r') as f:
    X_train = f['input_data'][:]

with h5py.File(dataset_path / 'output128x128_train.h5', 'r') as f:
    Y_train = f['output_data'][:]

with h5py.File(dataset_path / 'input128x128_val.h5'  , 'r') as f:
    X_val = f['input_data'][:]

with h5py.File(dataset_path / 'output128x128_val.h5', 'r') as f:
    Y_val = f['output_data'][:]

with h5py.File(dataset_path / 'input128x128_pred.h5', 'r') as f:
    X_test = f['input_data'][:]

with h5py.File(dataset_path / 'output128x128_pred.h5', 'r') as f:
    Y_test = f['output_data'][:]


# Shuffle the validation dataset once
# Get the total number of validation samples
num_val_samples = X_val.shape[0]

# Generate a random permutation of indices
np.random.seed(seed_for_shuffling)
indices = np.random.permutation(num_val_samples)

# Shuffle both arrays using the same indices
X_val_shuffled = X_val[indices]
Y_val_shuffled = Y_val[indices]

with tf.device('/CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_shuffled,Y_val_shuffled))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)

input_shape = X_train[0].shape
output_shape = Y_train[0].shape

# Take notes of parameters used
with open( results_folder / 'cmd_arguments.txt', 'w') as f:
    f.write(f'Model training job ID: {job_id}\n')
    f.write(f'Project Directory: {args.parent_folder}\n')
    f.write(f'Dataset seed: {args.dataset_seed}\n')
    f.write(f'Base batch size: {base_batch_size}\n')
    f.write(f'Epochs : {epochs}\n')
    f.write(f'Epoch loop num: {epoch_loop_num}\n')
    f.write(f'Initial learning rate: {initial_lr}\n')
    f.write(f'Target learning rate: {target_lr}\n')
    f.write(f'Learning rate decay weight: {lr_decay_weight}\n')
    f.write(f'Warm up step percentage: {warm_up_step_percentage}\n')
    f.write(f'Regularization weight: {reg_weight}\n')
    f.write(f'Random seed for tf:{seed_for_tf}\n')
    f.write(f'Random seed for validation dataset shuffling:{seed_for_shuffling}\n')

# Calculate warm up steps
total_epochs = epochs * epoch_loop_num
steps_per_epoch = int(np.ceil(X_train.shape[0] / global_batch_size))
total_steps = steps_per_epoch * total_epochs
warm_up_steps = int(math.floor(warm_up_step_percentage * total_steps))

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps:{total_steps}")
print(f"Warm up steps:{warm_up_steps}")

if warm_up_steps >= steps_per_epoch * epochs:
    raise ValueError("Too many warm up steps. We have only set up to do warm ups correctly in the first loop.")

#%%
#########################
# define loss
#########################

def normalized_mse(y_true, y_pred):
    '''
    The function calculates the normalized mse for each sample and returns a tensor with shape [batchsize,]
    Since we are both dividing by the same number of total pixels, we can do reduce_sum for both the numerator and denominator 
    to speed up a little bit.

    Otherwise, it is supposed to be:
    numerator = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3]) 
    denominator = tf.reduce_mean(tf.square(y_true), axis=[1, 2, 3]) 

    We don't do the average over one batch by ourself because in distributed training, we actually do batch average by global batch size.
    That is, the calculated average nmse per batch later on would be scaled down by 1/num_of_replicas compared to the true average nmse per batch.
    '''
    # Using normalized mse
    numerator = tf.reduce_sum(tf.square(y_true - y_pred), axis = [1, 2, 3]) # this is calculating the mean squared errors base on pixel difference per sample
    denominator = tf.reduce_sum(tf.square(y_true), axis = [1, 2, 3])        # this is calculating the mean squared values for the y_true results
    nmse_per_sample_tensor = numerator / denominator 
    return nmse_per_sample_tensor

#####################################
# define test set evaulation callback
#####################################
class test_data_evaluation(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch, logs=None):
        
        if logs is None:
            logs = {}

        results = self.model.evaluate(test_dataset, verbose=0, return_dict=True)
        logs["test_loss"] = results["loss"]

#########################
# Training 
#########################

# setup
start2 = time.time()

with strategy.scope():
    vae_model = create_vae(reg_weight = reg_weight, input_size = input_shape, output_channels = output_shape[-1])
    vae_model.summary()

    base_lr_var = tf.Variable(target_lr, dtype = tf.float32, trainable = False, name = "base_lr")
    lr_schedule = LinearWarmupToPiecewiseConstant(base_lr_var = base_lr_var, initial_lr = initial_lr, 
                                                  target_lr = target_lr, warmup_steps = warm_up_steps)
    
    # This is how we want to start our first loop 
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    vae_model.compile(optimizer = optimizer, loss = normalized_mse)

    callback1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    results_folder / 'model_best.weights.h5',
    monitor = "val_loss",
    save_best_only = True,
    save_weights_only = True   
    )

    callback2_test_eval = test_data_evaluation(test_dataset)

# training
print("Loop 0:") 
history = vae_model.fit(
    train_dataset,
    epochs = epochs, 
    callbacks = [callback1_checkpoint, callback2_test_eval],
    validation_data = val_dataset,
    verbose = 2,
)

train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
test_loss = np.array(history.history['test_loss'])
epochs_ran = history.epoch[-1] + 1 #+1 b/c epoch is zero indexing
filename = results_folder / 'history000.mat'
data = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss':test_loss, 'epochs_ran': epochs_ran}
savemat(filename, data)

current_lr = target_lr

for i in range(1,epoch_loop_num):

    if i % 2 == 0: # if i is even, then we change the learning rate
        current_lr *= lr_decay_weight

    # update the new learning rate
    lr_schedule.base_lr_var.assign(current_lr)
    lr_assigned = float(lr_schedule.base_lr_var.numpy())
    eff_lr = float(lr_schedule(optimizer.iterations).numpy())
    
    # This is for double-checking
    print(f"\nLoop {i}: lr variable assigned = {lr_assigned}, effective lr in optimizer={eff_lr}")
    print("-" * 40)

    history = vae_model.fit(
        train_dataset,
        epochs = epochs, 
        callbacks = [callback1_checkpoint, callback2_test_eval],
        validation_data = val_dataset,
        verbose = 2,
    )

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    test_loss = np.array(history.history['test_loss'])
    epochs_ran = history.epoch[-1] + 1 #+1 b/c epoch is zero indexing
    filename = results_folder / f'history{i}00.mat'
    data = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss, 'epochs_ran': epochs_ran}
    savemat(filename, data)

end2 = time.time()

#########################
# Training time calculation
#########################
elapsed2 = end2 - start2 
print(f"Time used for training: {elapsed2:.2f} seconds")
