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
parser.add_argument('-l', '--initial_learning_rate', type=float)
parser.add_argument('-tl', '--target_learning_rate', type=float)
parser.add_argument('-el','--end_learning_rate',type=float)
parser.add_argument('-w', '--warm_up_step_percentage', type=float)
parser.add_argument('-r', '--reg_weight',type=float)
parser.add_argument('-tfs','--tf_seed', type=int)
args = parser.parse_args()

base_batch_size = args.base_batch_size
total_epochs = args.epoch
initial_lr = args.initial_learning_rate
target_lr = args.target_learning_rate
end_lr = args.end_learning_rate
reg_weight = args.reg_weight
global_batch_size = base_batch_size * num_GPUs
seed_for_tf = args.tf_seed
warm_up_step_percentage = args.warm_up_step_percentage

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
results_folder = project_path / f'b{base_batch_size}_e{total_epochs}_lr{initial_lr}_r{reg_weight}'/ f'job_{job_id}'
results_folder.mkdir(parents=True, exist_ok=True)

plotting_folder = results_folder / 'plotting_info'
plotting_folder.mkdir(parents=True, exist_ok=True)

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


# Generate a random permutation of indices
seed_for_shuffling = 283
num_val_samples = X_val.shape[0]
np.random.seed(seed_for_shuffling)
indices = np.random.permutation(num_val_samples)

# Shuffle both arrays using the same indices
X_val_shuffled = X_val[indices]
Y_val_shuffled = Y_val[indices]

with tf.device('/CPU:0'):
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_shuffled,Y_val_shuffled))

buffer_size = 2048
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(global_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(global_batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

input_shape = X_train[0].shape
output_shape = Y_train[0].shape

# Take notes of parameters used
with open( results_folder / 'cmd_arguments.txt', 'w') as f:
    f.write(f'Model training job ID: {job_id}\n')
    f.write(f'Project Directory: {args.parent_folder}\n')
    f.write(f'Dataset seed: {args.dataset_seed}\n')
    f.write(f'Base batch size: {base_batch_size}\n')
    f.write(f'Epochs : {total_epochs}\n')
    f.write(f'Initial learning rate: {initial_lr}\n')
    f.write(f'Target learning rate: {target_lr}\n')
    f.write(f'End learning rate: {end_lr}\n')
    f.write(f'Warm up step percentage: {warm_up_step_percentage}\n')
    f.write(f'Regularization weight: {reg_weight}\n')
    f.write(f'Random seed for tf:{seed_for_tf}\n')

# Calculate warm up steps
steps_per_epoch = int(X_train.shape[0]//global_batch_size)
total_steps = steps_per_epoch * total_epochs
warm_up_steps = int(math.floor(warm_up_step_percentage * total_steps))
decay_steps = total_steps - warm_up_steps
val_steps = int(X_val.shape[0] // global_batch_size)

if warm_up_steps >= total_steps:
    raise ValueError("Warm up steps can't be larger than total number of steps. Check warm up step percentage.")

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total steps:{total_steps}")
print(f"Warm up steps:{warm_up_steps}")
print(f"Decay steps:{decay_steps}")
print(f"Val steps:{val_steps}")

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
# Define custom callbacks
#####################################
class val_loss_monitor_weight_saver(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, min_delta = 0.001, patience = 20):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.min_delta = min_delta
        self.patience = patience
        self.best_val_loss = float("inf")
        self.epochs_plateaued = 0
        self.best_weights = None
        self.best_epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs["val_loss"]
        # If improved more than min_delta, current val loss become new loss
        # Plateaued count resets back to 0
        if current_val_loss < self.best_val_loss - self.min_delta: 
            self.best_val_loss = current_val_loss
            self.epochs_plateaued = 0
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        else:
            self.epochs_plateaued += 1

        if self.epochs_plateaued >= self.patience:
            save_path = self.save_dir / f"vae_epoch_{self.best_epoch:04d}.weights.h5"
            print(f"\nWeight at epoch {self.best_epoch} saved.")
            current_weight = self.model.get_weights()

            self.model.set_weights(self.best_weights)
            self.model.save_weights(save_path)

            self.model.set_weights(current_weight)

            lr = self.model.optimizer.learning_rate
            if callable(lr):
                lr = lr(self.model.optimizer.iterations)
            
            print(f"\nCurrent learning rate is {lr.numpy()}")


#########################
# Training 
#########################

# setup
start2 = time.time()

with strategy.scope():
    vae_model = create_vae(reg_weight = reg_weight, input_size = input_shape, output_channels = output_shape[-1])
    vae_model.summary()

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps, alpha=end_lr, name="LinearWarmupCosineDecay", 
                                                            warmup_target=target_lr, warmup_steps=warm_up_steps)
    
    # This is how we want to start our first loop 
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
    vae_model.compile(optimizer = optimizer, loss = normalized_mse)

    callback1_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    results_folder / 'model_best.weights.h5',
    monitor = "val_loss",
    save_best_only = True,
    save_weights_only = True   
    )

    callback2_epoch_weight_saver = val_loss_monitor_weight_saver(save_dir = results_folder)


history = vae_model.fit(
    train_dataset,
    epochs = total_epochs, 
    callbacks = [callback1_checkpoint, 
                 callback2_epoch_weight_saver],
    validation_data = val_dataset,
    validation_steps = val_steps, 
    verbose = 2,
    steps_per_epoch = steps_per_epoch
)

train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])
epochs_ran = history.epoch[-1] + 1 #+1 b/c epoch is zero indexing
filename = results_folder / 'history.mat'
data = {'train_loss': train_loss, 'val_loss': val_loss, 'epochs_ran': epochs_ran}
savemat(filename, data)

np.save(plotting_folder / 'train_loss.npy', train_loss)
np.save(plotting_folder / 'val_loss.npy', val_loss)
print(f"Epochs ran in the end:{epochs_ran}")

# Save full final model and final weights
vae_model.save(results_folder / "model_final_snapshot.keras")
vae_model.save_weights(results_folder/"model_final_snapshot.weights.h5")

end2 = time.time()

#########################
# Training time calculation
#########################
elapsed2 = end2 - start2 
print(f"Time used for training: {elapsed2:.2f} seconds")
