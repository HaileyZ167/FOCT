import torch
import numpy as np
import cv2
import os
from torch import nn, optim
import torch.nn.init as init
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
class CT_Net(nn.Module):
    def __init__(self, hidden=512, dim=256):
        super(CT_Net, self).__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)

        return self.fc3(x)

    def reset_params(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

class CT_metrics_system(nn.Module):
    def __init__(self, rho):
        super(CT_metrics_system, self).__init__()
        self.Epoch = 100
        self.net = CT_Net(hidden=512, dim=1024).cuda()

        self.rho = rho
        self.rho_raw = nn.Parameter(torch.randn(1))
        # 混合精度
        self.scaler = GradScaler()
        self.mse_loss = nn.MSELoss()

    def forward(self, M_aug, Q, N_fg):
        """
        Args:
            M_aug: M_org (+ selected query features)
            Q: current query features
            N_fg: number of current foreground features

        Returns:
            score: anomaly scores
            backward_map: backward map of fg_c and query

        """
        optimizer = optim.AdamW(self.net.parameters(), lr=2e-4)

        self.net.reset_params()
        for epoch in range(self.Epoch):
            self.net.train()

            rho = torch.sigmoid(self.rho_raw).cuda()
            with torch.enable_grad():
                optimizer.zero_grad()

                with autocast():
                    mse_n = (M_aug[:, None] - Q).pow(2)
                    cost = mse_n.sum(-1)
                    d = self.net(mse_n).squeeze().mul(-1)  # navigator distance: B x B  与-1相乘
                    forward_map = torch.softmax(d, dim=1)  # forward map is in y wise
                    backward_map = torch.softmax(d, dim=0)

                    ct_loss = self.rho * (cost * forward_map).sum(1).mean() + (1 - self.rho) * (
                            cost * backward_map).sum(0).mean()
                    # print("ct_metric_loss:", ct_loss)
                # AMP
                self.scaler.scale(ct_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()


        with torch.no_grad():
            # CT_metrics CT(Sc(fg+bg+q), Q)
            mse_aug = (M_aug[:, None] - Q).pow(2)
            cost = mse_aug.sum(-1)
            d_ = self.net(mse_aug).squeeze().mul(-1)
            backward_map_aug = torch.softmax(d_, dim=0)
            score = (cost * backward_map_aug).sum(0)
            scoremap_fg = (cost * backward_map_aug)[:N_fg, :]
            score_fg = scoremap_fg.sum(0)

        # return score
        return score, score_fg

class CT_AUG_system(nn.Module):
    def __init__(self, rho):
        super(CT_AUG_system, self).__init__()
        self.Epoch = 100
        self.net = CT_Net(hidden=512, dim=1024).cuda()

        self.rho = rho
        self.rho_raw = nn.Parameter(torch.randn(1))
        # 混合精度
        self.scaler = GradScaler()
        self.mse_loss = nn.MSELoss()

    def forward(self, M_org, Q):
        """
        Args:
            M_org: foreground coreset features + background coreset features
            Q: query features

        Returns:
            score: anomaly scores without augmented features
            backward_map: backward map of M_org and query
        """
        optimizer = optim.AdamW(self.net.parameters(), lr=2e-4)

        self.net.reset_params()
        for epoch in range(self.Epoch):
            self.net.train()

            rho = torch.sigmoid(self.rho_raw).cuda()
            with torch.enable_grad():
                optimizer.zero_grad()

                with autocast():
                    mse_n = (M_org[:, None] - Q).pow(2)
                    cost = mse_n.sum(-1)
                    d = self.net(mse_n).squeeze().mul(-1)  # navigator distance: B x B  与-1相乘
                    forward_map = torch.softmax(d, dim=1)  # forward map is in y wise
                    backward_map = torch.softmax(d, dim=0)

                    ct_loss = self.rho * (cost * forward_map).sum(1).mean() + (1 - self.rho) * (
                            cost * backward_map).sum(0).mean()
                    # print("ct_aug_loss:", ct_loss)
                # AMP
                self.scaler.scale(ct_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

        with torch.no_grad():
            # CT_metrics CT(Sc(fg+bg+q), Q)
            mse_org = (M_org[:, None] - Q).pow(2)
            cost = mse_org.sum(-1)
            d_ = self.net(mse_org).squeeze().mul(-1)
            backward_map_org = torch.softmax(d_, dim=0)
            score_org = (cost * backward_map_org).sum(0)

        return score_org, backward_map_org