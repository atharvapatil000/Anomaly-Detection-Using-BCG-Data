# -*- coding: utf-8 -*-
"""movementvalues.ipynb


"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('/icapotech/ground_truth/Subject39_1526417507/1526417507.csv', header = None)
data = data.T
data2 = pd.read_csv('/icapotech/ground_truth/Subject39_1526591202/1526591202.csv', header = None)
data2 = data2.T
data3 = pd.read_csv('/icapotech/ground_truth/Subject42_1527280030/1527280030.csv', header = None)
data3 = data3.T
data4 = pd.read_csv('/icapotech/ground_truth/Subject43_1527806941/1527806941.csv', header = None)
data4 = data4.T
data5 = pd.read_csv('/icapotech/ground_truth/Subject54_1539288817/1539288817.csv', header = None)
data5 = data5.T
data6 = pd.read_csv('/icapotech/ground_truth/Subject55_1539459892/1539459892.csv', header = None)
data6 = data6.T
train_data = pd.concat([data,data2,data3,data4,data5,data6],ignore_index = True)

data = pd.read_csv('/icapotech/ground_truth/Subject39_1526417507/movementvalues.csv', header = None)
data = data.T
data = data.drop(0)
data2 = pd.read_csv('/icapotech/ground_truth/Subject39_1526591202/movementvalues.csv', header = None)
data2 = data2.T
data2 = data2.drop(0)
data3 = pd.read_csv('/icapotech/ground_truth/Subject42_1527280030/movementvalues.csv', header = None)
data3 = data3.T
data3 = data3.drop(0)
data4 = pd.read_csv('/icapotech/ground_truth/Subject43_1527806941/movementvalues.csv', header = None)
data4 = data4.T
data4 = data4.drop(0)
data5 = pd.read_csv('/icapotech/ground_truth/Subject54_1539288817/movementvalues.csv', header = None)
data5 = data5.T
data5 = data5.drop(0)
data6 = pd.read_csv('/icapotech/ground_truth/Subject55_1539459892/movementvalues.csv', header = None)
data6 = data6.T
data6 = data6.drop(0)
target_values = pd.concat([data,data2,data3,data4,data5,data6],ignore_index = True)

class LSTM(nn.Module):
  def __init__(self, n_hidden = 512):
    super(LSTM, self).__init__()
    self.n_hidden = n_hidden
    self.lstm = nn.LSTM(input_size = 226, hidden_size = 512, batch_first = True)
    self.linear = nn.Linear(self.n_hidden, 3)

  def forward(self, x):
    h_t, c_t = self.lstm(x)
    h_t = h_t.squeeze()
    res = self.linear(h_t)
    res = res.T
    return res

class MovementDataset(torch.utils.data.Dataset):
  def __init__(self, train_data, targets = None):
    self.train_data = train_data
    self.targets = targets

  def __len__(self):
    return len(self.train_data)

  def __getitem__(self, idx):
    if self.targets is not None:
      try:
        train, targ = self.train_data.loc[idx], self.targets.loc[idx]
      except:
        train, targ = self.train_data.iloc[idx], self.targets.iloc[idx]
      train = train.values.reshape((1,len(train)))
      targ = targ.values.reshape((1,len(targ)))
      train = train.astype(np.float32)
      targ = targ.astype(np.float32)
      train = torch.tensor(train)
      targ = torch.tensor(targ)
      train = train.view((1, 240, 226))
      return train, targ
    else:
      try:
        train = self.train_data.loc[idx]
      except:
        train = self.train_data.iloc[idx]
      train = train.values.reshape((1,len(train)))
      train = train.astype(np.float32)
      train = torch.tensor(train)
      train = train.view((1, 240, 226))
      return train

traindata,validation =  train_test_split(train_data, test_size=0.16)

train_dataset = MovementDataset(traindata,target_values)
val_dataset = MovementDataset(validation, target_values)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

def train(model, lossfn, optimizer, scheduler, device, epochs = 250):
    losses = []
    model.to(device)
    for i in range(epochs):
        model.train()
        print(f"Currently running epoch number {i+1}")
        for t, (xb, yb) in enumerate(train_dataloader):
            xb = xb.to(device, dtype = torch.float32)
            yb = yb.to(device, dtype = torch.int64)
            xb = xb.squeeze(0)
            yb = yb.squeeze()
            predictions = model(xb)
            predictions = predictions.T
            predictions = predictions.to(device, dtype = torch.float32)
            loss = lossfn(predictions, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = torch.sum(torch.argmax(predictions, dim = 1) == yb)/len(yb)
        print(f"accuracy for train is {acc}")
        scheduler.step(loss)
        #print(f"Train loss for this epoch is {loss}")
        del loss,predictions

        with torch.no_grad():
          model.eval()
          for v, (xv, yv) in enumerate(val_dataloader):
            xv = xv.to(device, dtype = torch.float32)
            yv = yv.to(device, dtype = torch.int64)
            xv = xv.squeeze(0)
            yv = yv.squeeze()
            predictions = model(xv)
            predictions = predictions.T
            predictions = predictions.to(device, dtype = torch.float32)
            loss = lossfn(predictions, yv)
            acc = torch.sum(torch.argmax(predictions, dim = 1) == yv)/len(yv)
            print(f"accuracy for val is {acc}")
          #print(f"Val Loss for this epoch is {loss}")

model = LSTM()
lossfn = torch.nn.functional.cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 7, verbose = True, eps = 1e-11)

train(model, lossfn, optimizer, scheduler, device)

import os

df = pd.DataFrame()
addresses = []
for root, dir, files in os.walk('/icapotech/dataset'):
  if files == []:
    pass
  else:
    path = os.path.join(root, files[0])
    data = pd.read_csv(path, header = None)
    addresses.append(root)
    data = data.T
    df = pd.concat([df,data], ignore_index = True)

test_dataset = MovementDataset(df)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True)

with torch.no_grad():
        model.eval()
        for v, xv in enumerate(test_dataloader):
          xv = xv.to(device, dtype = torch.float32)
          xv = xv.squeeze(0)
          predictions = model(xv)
          predictions = torch.argmax(predictions, dim = 0)
          predictions = predictions.detach().cpu().numpy()
          path = addresses[v]
          time = int(path[-10:])
          timestamps = np.arange(time, time+240)
          predictionsdf = pd.DataFrame([timestamps,predictions])
          predictionsdf = predictionsdf.transpose()
          submission = predictionsdf.to_csv(os.path.join(path,'movementvalues.csv'), index = False, header = None)



