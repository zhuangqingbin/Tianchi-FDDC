#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:08:21 2018

@author: jimmy
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# replace following class code with an easy sequential network
class Net(torch.nn.Module):
    def __init__(self,inchanels):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(inchanels, 2000)   # hidden layer
        self.hidden2 = torch.nn.Linear(2000, 3000)
        self.hidden3 = torch.nn.Linear(3000, 2500)
        self.hidden4 = torch.nn.Linear(2500, 2000)
        self.hidden5 = torch.nn.Linear(2000, 1500)
        self.hidden6 = torch.nn.Linear(1500,1000)
        self.hidden7 = torch.nn.Linear(1000,800)
        self.hidden8 = torch.nn.Linear(800,400)
        self.hidden9 = torch.nn.Linear(400,100)
        self.hidden10 = torch.nn.Linear(100,10)
        self.hidden11 = torch.nn.Linear(10,1)


    def forward(self, x):
        x = F.relu(self.hidden1(x))      
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden10(x))
 
        x = self.hidden11(x)       
        return x



def train_model(model, criterion, optimizer, data, num_epochs):
    model.train()
    data_loader = DataLoader(
            TensorDataset(torch.from_numpy(data[:,4:]).float(),
                torch.from_numpy(data[:,3:4]).float()),
            batch_size=int(np.ceil(len(data)/2)), 
            shuffle=True, num_workers=4
    )
            
    for epoch in range(num_epochs):    
        for tr_x, tr_y in data_loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(tr_x)
                loss = criterion(outputs, tr_y)

                loss.backward()
                optimizer.step()
    return model




def pre(data,num_epochs,scale=True):
    df = data.copy().dropna(axis='columns',how='all')
    
    cond = (df.END_DATE==2018)&(df.FISCAL_PERIOD==2)
    df_tr, df_te = df[~cond].values, df[cond].values
    te_x = df_te[:,4:] 
    
    
    net = Net(len(df.columns)-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model = train_model(model=net, criterion=criterion, optimizer=optimizer, data=df_tr, num_epochs=num_epochs)
    return model(torch.from_numpy(df_te[:,4:]).float()).data.numpy()[0,0]





def RevenuePre(data):
    """ 营业收入预测 """
    df = data.copy()

    rev_model = df.groupby('TICKER_SYMBOL',as_index=True).apply(pre,200)
    df = pd.DataFrame(rev_model,columns=['predicted'])
    df.reset_index(drop=False,inplace=True)
    return df
