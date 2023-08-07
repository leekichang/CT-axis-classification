import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

import utils

class SupervisedTrainer(object):
    def __init__(self):
        self.save_path    = f'./checkpoints/'
        os.makedirs(self.save_path, exist_ok=True)
        self.epoch        = 0
        self.epochs       = 2
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model        = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 2)
        self.model.to(self.device)
        self.criterion    = nn.CrossEntropyLoss()
        self.optimizer    = optim.AdamW(self.model.parameters(), lr=0.0001)
        
        self.trainset     = utils.load_dataset(is_train=True)
        self.testset      = utils.load_dataset(is_train=False)

        self.train_loader = DataLoader(self.trainset, batch_size=128, shuffle=True , drop_last=True )
        self.test_loader  = DataLoader(self.testset , batch_size=128, shuffle=False, drop_last=False)
    
        self.train_loss = []
        self.test_loss  = []
        self.accs       = []

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'model name:resnet\ndataset:axial-coronal\ndevice:{self.device}\nTotal parameter:{total_params:,}')

    def train(self):
        loss_trace = []
        self.model.train()
        for _, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            loss_trace.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()
        self.train_loss.append(np.average(loss_trace))
    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        loss_trace = []
        result_pred, result_anno = [], []
        for idx, batch in enumerate(self.test_loader):
            X, Y = batch
            X, Y = X.to(self.device), Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            loss_trace.append(loss.cpu().detach().numpy())
            pred_np  = pred.to('cpu').detach().numpy()
            pred_np  = np.argmax(pred_np, axis=1).squeeze()
            Y_np     = Y.to('cpu').detach().numpy().reshape(-1, 1).squeeze()
            result_pred = np.hstack((result_pred, pred_np))
            result_anno = np.hstack((result_anno, Y_np))
        acc = metrics.accuracy_score(y_true=result_anno, y_pred=result_pred)
        self.test_loss.append(np.average(loss_trace))
        self.accs.append(acc)
        # self.TB_WRITER.add_scalar(f'Test Loss', np.average(loss_trace), self.epoch+1)
        # self.TB_WRITER.add_scalar(f'Test Accuracy', acc, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss[self.epoch]:>6.4f} Test Loss:{self.test_loss[self.epoch]:>6.4f} Test Accuracy:{self.accs[self.epoch]*100:>5.2f}%')
        
        
if __name__ == '__main__':
    torch.manual_seed(0)
    trainer = SupervisedTrainer()
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        #if (trainer.epoch+1)%10 == 0:
        trainer.save_model()
        trainer.epoch += 1