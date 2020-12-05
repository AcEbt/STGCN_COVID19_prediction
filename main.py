import os
import numpy as np
import torch 
import torch.nn as nn
import Data2Graph
#import matplotlib.pyplot as plt
import pickle as pk

from stgcn import STGCN
from utils import generate_dataset, get_normalized_adj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_filename='us-counties.csv'
auxiliary_filename='uszips.csv'

feat_tensor, Adj, active_cases, confirmed_cases, popn, selected_counties,g = Data2Graph.load_data(data_filename, auxiliary_filename, active_thresh=1000)
cases = torch.stack([active_cases,confirmed_cases],dim=2)
all_in = torch.cat([feat_tensor,cases],dim=2)
np.save('Adj',Adj.numpy())
np.save('cases',cases.numpy())
np.save('all_in',all_in.numpy())

def train_epoch(training_input, training_target, batch_size):

    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


num_timesteps_input = 15
num_timesteps_output = 5

epochs = 200
batch_size = 20
lr = 1e-4
torch.manual_seed(1)

A = np.load("Adj.npy")
X = np.load("cases.npy").transpose((0, 2, 1))

X = X.astype(np.float32)
X[:, -2, :] = X[:, -2, :] / 100000
X[:, -1, :] = X[:, -1, :] / 10000

split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]

training_input, training_target = generate_dataset(train_original_data,
                                                   num_timesteps_input=num_timesteps_input,
                                                   num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                         num_timesteps_input=num_timesteps_input,
                                         num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                           num_timesteps_input=num_timesteps_input,
                                           num_timesteps_output=num_timesteps_output)

A_wave = get_normalized_adj(A)
A_wave = torch.from_numpy(A_wave)

net = STGCN(A_wave.shape[0],
            training_input.shape[3],
            num_timesteps_input,
            num_timesteps_output)

training_losses = []
validation_losses = []
validation_maes = []

loss_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
for epoch in range(epochs + 1):
    if epoch > 50:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr / 10)

    loss = train_epoch(training_input, training_target,
                       batch_size=batch_size)
    training_losses.append(loss)

    # Run validation
    with torch.no_grad():
        net.eval()

        out = net(A_wave, val_input)
        val_loss = loss_criterion(out, val_target).to(device="cpu")
        validation_losses.append(val_loss.detach().numpy().item())

        out_unnormalized = out.detach().cpu().numpy()
        target_unnormalized = val_target.detach().cpu().numpy()
        mae = np.mean(np.absolute(out_unnormalized[:, :, -2] - target_unnormalized[:, :, -2]))

        validation_maes.append(mae)
        out = None
        val_input = val_input.to(device="cpu")
        val_target = val_target.to(device="cpu")

    if epoch % 20 == 0:
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAPE: {}".format(validation_maes[-1]))
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.legend()
        plt.show()

#    checkpoint_path = "checkpoints/"
#    if not os.path.exists(checkpoint_path):
#        os.makedirs(checkpoint_path)
#    with open("checkpoints/losses.pk", "wb") as fd:
#        pk.dump((training_losses, validation_losses, validation_maes), fd)


with torch.no_grad():
    net.eval()

    out = net(A_wave, training_input)
    trainging_loss = loss_criterion(out, training_target).to(device="cpu")

    out_unnormalized = out.detach().cpu().numpy()
    target_unnormalized = training_target.detach().cpu().numpy()
    '''
    mape = np.mean(np.absolute(out_unnormalized[:, :, -2] - target_unnormalized[:, :, -2])
                           / target_unnormalized[:, :, -2])
    validation_maes.append(mape)
    '''
    mae = np.mean(np.absolute(out_unnormalized[:, :, -2] - target_unnormalized[:, :, -2])
                  )
    out = None
    test_input = training_input.to(device="cpu")
    test_target = training_target.to(device="cpu")


county_num = 15
prediction = out_unnormalized[:,county_num,-2]
true_value = target_unnormalized[:,county_num,-2]

plt.plot(prediction, label="Prediction")
plt.plot(true_value, label="Groud Truth")
plt.legend()
plt.show()


with torch.no_grad():
    net.eval()

    out = net(A_wave, test_input)
    test_loss = loss_criterion(out, test_target).to(device="cpu")

    out_unnormalized = out.detach().cpu().numpy()
    target_unnormalized = val_target.detach().cpu().numpy()

    '''
    mape = np.mean(np.absolute(out_unnormalized[:, :, -2] - target_unnormalized[:, :, -2])
                    / target_unnormalized[:, :, -2])
    validation_maes.append(mape)
    '''
    mae = np.mean(np.absolute(out_unnormalized[:, :, -2] - target_unnormalized[:, :, -2])
                  )
    out = None
    test_input = test_input.to(device="cpu")
    test_target = test_target.to(device="cpu")

print("MAPE: {}".format(mae))
print("Test loss: {}".format(test_loss.detach().numpy().item()))

county_num = 15
prediction = out_unnormalized[:,county_num,-2]
true_value = target_unnormalized[:,county_num,-2]

plt.plot(prediction, label="Prediction")
plt.plot(true_value, label="Groud Truth")
plt.legend()
plt.show()