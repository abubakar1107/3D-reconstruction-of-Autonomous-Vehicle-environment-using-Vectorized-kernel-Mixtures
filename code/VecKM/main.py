import time
import sys
import importlib
import torch
import torch.nn.functional as F
import numpy as np
from VecKM import NormalEstimator
import os

def get_filenames(path):
    with open(path, 'r') as f:
        filenames = f.readlines()
        filenames = [x.strip() for x in filenames]
    return filenames
    
def get_dataset(filenames):
    pts_list, normal_list = [], []
    for filename in filenames:
        pts = np.loadtxt(f'PCPNet/{filename}.xyz')
        normal = np.loadtxt(f'PCPNet/{filename}.normals')
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        pts = pts / np.max(np.linalg.norm(pts, axis=1))
        pts_list.append(torch.from_numpy(pts).float())
        normal_list.append(torch.from_numpy(normal).float())
    return pts_list, normal_list

def random_rotate(pts, normal):
    alpha, beta, gamma = np.random.rand(3) * 2 * np.pi
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = torch.tensor([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = torch.tensor([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    R = (Rx @ Ry @ Rz).float().cuda()
    pts = pts @ R.T
    normal = normal @ R.T
    return pts, normal

train_files_list = get_filenames('PCPNet/list/trainingset_whitenoise.txt')
train_pts_list, train_normal_list = get_dataset(train_files_list)
test_filenames = [
    'PCPNet/list/testset_no_noise.txt',
    'PCPNet/list/testset_low_noise.txt',
    'PCPNet/list/testset_med_noise.txt',
    'PCPNet/list/testset_high_noise.txt',
    'PCPNet/list/testset_vardensity_gradient.txt',
    'PCPNet/list/testset_vardensity_striped.txt', 
]
test_data_all = []
for filename in test_filenames:
    test_files_list = get_filenames(filename)
    test_pts_list, test_normal_list = get_dataset(test_files_list)
    test_data_all.append((test_pts_list, test_normal_list))
    
model = NormalEstimator().cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss = [100] * 6
for epoch in range(9999999):
    train_total_time, train_total_loss = 0, 0
    model = model.train()
    for pts, normal in zip(train_pts_list, train_normal_list):
        # data augmentation
        pts, normal = pts.cuda(), normal.cuda()
        pts, normal = random_rotate(pts, normal)

        # training.
        start = time.time()
        pred_normal, gt_normal = model(pts, normal)
        cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1).abs()
        loss = -cos_sim.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        end = time.time()
        train_total_time += end - start
        train_total_loss += loss.item()

        # training loss computation.
        cos_sim[cos_sim>1] = 1
        angle = torch.acos(cos_sim) * 180 / np.pi
        rmse = torch.sqrt((angle ** 2).mean())
        train_total_loss += rmse.item()
        
    count = len(train_pts_list)
    print(f'Epoch {epoch}. train loss: {train_total_loss/count}; train time: {train_total_time}s for {count} shapes.')

    if (epoch+1) % 100 == 0:
        model = model.eval()
        for i, (test_pts_list, test_normal_list) in enumerate(test_data_all):
            test_total_time, test_total_loss = 0, 0
            for pts, normal in zip(test_pts_list, test_normal_list):
                pts, normal = pts.cuda(), normal.cuda()

                with torch.no_grad():
                    start = time.time()
                    pred_normal, gt_normal = model(pts, normal)
                    test_total_time += time.time() - start

                cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=1).abs()
                cos_sim[cos_sim>1] = 1
                angle = torch.acos(cos_sim) * 180 / np.pi
                rmse = torch.sqrt((angle ** 2).mean())
                test_total_loss += rmse.item()

            count = len(test_pts_list)
            print(f'\ttest loss: {test_total_loss/count}; test time: {test_total_time} for {count} shapes.')

            if test_total_loss/count < best_loss[i]:
                best_loss[i] = test_total_loss/count
                torch.save(model.state_dict(), f'log/best_model_{i}.pth')
                print(f'\tbest model saved to log/best_model_{i}.pth')

        print(f'best loss: {best_loss}')