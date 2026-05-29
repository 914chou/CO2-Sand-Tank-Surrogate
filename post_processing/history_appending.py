from scipy.io import savemat
from scipy.io import loadmat
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-p','--parent_folder')
parser.add_argument('-b','--base_batch_size', type=int)
parser.add_argument('-e','--epoch', type=int)
parser.add_argument('-el','--epoch_loop',type=int)
parser.add_argument('-l','--learning_rate',type=float)
parser.add_argument('-r','--reg_weight',type=float)
parser.add_argument('-j','--job_id')
parser.add_argument('-fl','--final_loop_num',type=int)
args = parser.parse_args()

base_batch_size = args.base_batch_size
epochs = args.epoch
epoch_loop_num = args.epoch_loop
initial_lr = args.learning_rate
reg_weight = args.reg_weight
job_id = args.job_id
final_loop_num = args.final_loop_num

#%%
##############################################
# Set up project folder
##############################################
project_path = Path(args.parent_folder)

# This is the folder where the trained weights and history files are located from model_training.py
results_folder = project_path / f'b{base_batch_size}_e{epochs * epoch_loop_num}_lr{initial_lr}_r{reg_weight}/'/ f'job_{job_id}'

# Subfolder for things to visualize
plotting_folder = results_folder / 'plotting_info'
plotting_folder.mkdir(parents=True, exist_ok=True)

# Concatentate the losses from all runs together so we can plot results
train_loss_list = []
val_loss_list = []
test_loss_list = []
epoch_list = []

for i in range(final_loop_num):
    filename = results_folder / f'history{i}00.mat'
    history_info = loadmat(filename)
    train_loss = np.array(history_info['train_loss']) 
    val_loss = np.array(history_info['val_loss'])
    test_loss = np.array(history_info['test_loss'])

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    test_loss_list.append(test_loss)

    # Also obtain the info of the number of actual epochs ran per loop
    epoch_list.append(history_info['epochs_ran'].item())
   
combined_train_loss = np.concatenate(train_loss_list, axis = 1)
combined_val_loss = np.concatenate(val_loss_list, axis = 1)
combined_test_loss = np.concatenate(test_loss_list, axis = 1)

final_train_loss = train_loss[:,-1] 
final_val_loss = val_loss[:,-1]
final_test_loss = test_loss[:,-1]
total_epochs = sum(epoch_list)

print("\n")
print("=" * 40)
print("\n")
print(f"Final training loss: {final_train_loss.item():.3e}")
print(f"Final validation loss: {final_val_loss.item():.3e}")
print(f"Final test loss: {final_test_loss.item():.3e}")
print(f"Total epochs for the training: {total_epochs}")
print("\n")

np.save(plotting_folder / 'train_loss.npy', combined_train_loss)
np.save(plotting_folder / 'val_loss.npy', combined_val_loss)
np.save(plotting_folder / 'test_loss.npy', combined_test_loss)
