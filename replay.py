import random
import matplotlib.pyplot as plt
import time
import copy

import numpy as np 
import torch
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.cluster import KMeans

#从每个类别的训练数据中随机抽取样本构建回放缓冲区。基线方法，简单高效但无法保留数据分布特征。
def update_random(train_per_class, partition, task, size, replay_buffer, total_data):
    for tasks in range(task+1):
        for i in partition[tasks]:
            proportion = min(int(len(train_per_class[i]) / total_data * size), len(train_per_class[i]))
            memo = random.sample(range(len(train_per_class[i])), k = proportion)
            memory = [train_per_class[i][idx] for idx in memo]
            memory = torch.from_numpy(np.array(memory))
            if replay_buffer == None:
                replay_buffer = memory
            else:
                replay_buffer = torch.cat((replay_buffer, memory),0)
    return replay_buffer

#选择能覆盖同类样本最多区域的节点，最大化类内多样性
def coverage_max(network, edge_index, train_per_class, size, replay, total_data, partition, task, features, distance):
    
    embeds = network.encode(features, edge_index)
    
    replay_buffer = None

    for cla in partition[task]:
        proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))

        count = 0
        memory = []
        temp_memory = []
        cover = []
        
        dist_matrix = torch.cdist(embeds[train_per_class[cla]], embeds[train_per_class[cla]], p=2)
        distances_mean = torch.mean(dist_matrix)
        dist_bin_matrix = torch.where(dist_matrix<distances_mean.item()*distance,1,0)

        temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
        while count < proportion:
            ind = (torch.sum(temp_dist_bin_matrix,0)).argmax()
            memory.append(train_per_class[cla][ind])
            temp_memory.append(ind)
            target_dist_matrix = temp_dist_bin_matrix[ind,:]
            new_cover = torch.where(target_dist_matrix==1)[0]
            cover = list(set(cover) | set(new_cover))
            temp_dist_bin_matrix[new_cover,:] = 0
            temp_dist_bin_matrix[:,new_cover] = 0

            #reset
            if len(cover) >= len(train_per_class[cla]) * 0.9:
                cover = temp_memory
                temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
                temp_dist_bin_matrix[cover,:] = 0
                temp_dist_bin_matrix[:,cover] = 0

            count += 1
        
        memory = torch.from_numpy(np.array(memory))

        replay[cla] = memory

    
    for i in range(task+1):
        for cla in partition[i]:
            proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))
            
            if replay_buffer == None:
                replay_buffer = replay[cla][:proportion]
            else:
                replay_buffer = torch.cat((replay_buffer, replay[cla][:proportion]), 0)

    return replay_buffer, replay

#选择靠近类别中心的样本（基于特征或嵌入均值)
def update_MF(network, edge_index, type, size, replay_buffer, total_data, partition, task, labels, train, train_per_class, features, device, mean_features, count, dist):

    if type == 'embedding':
        embeds = network.encode(features, edge_index)


    for cla in partition[task]:
        if type == 'feature':
            mean_features[cla] = torch.zeros(len(features[0])).to(device)
        elif type == 'embedding':
            mean_features[cla] = torch.zeros(embeds.size(1)).to(device)
        count[cla] = []
        dist[cla] = []

    for cla in partition[task]:
        if type == 'feature':
            mean_features[cla] = torch.mean(features[train_per_class[cla]], dim=0)
        elif type == 'embedding':
            mean_features[cla] = torch.mean(embeds[train_per_class[cla]], dim=0)

    for cla in partition[task]:
        if type == 'feature':
            dist[cla] = torch.cdist(mean_features[cla].reshape(-1, mean_features[cla].size(0)), features[train_per_class[cla]], p=2)[0]
        elif type == 'embedding':
            dist[cla] = torch.cdist(mean_features[cla].reshape(-1, mean_features[cla].size(0)), embeds[train_per_class[cla]], p=2)[0]

    for cla in mean_features.keys():
        proportion = min(int(len(train_per_class[cla]) / total_data * size), len(train_per_class[cla]))
        v, i = dist[cla].topk(proportion)
        memory = [train_per_class[cla][item] for item in i]
        memory = torch.from_numpy(np.array(memory))
        if replay_buffer == None:
            replay_buffer = memory
        else:
            replay_buffer = torch.cat((replay_buffer, memory), 0)

    return replay_buffer, mean_features, count, dist

#对每个样本，计算与所有其他类别样本的距离。
# 统计距离小于阈值的异类样本数，数量越少表示越靠近类中心。
# 选择异类干扰最少的样本（低计数）。
# 适用场景：保留类别边界信息，防止类间混淆。
def update_CM(network, edge_index, type, train_per_class, size, replay_buffer, total_data, partition, task, features, cm, distance):
    
    if type == 'embedding':
        embeds = network.encode(features, edge_index)

    for cla in partition[task]:
        cm[cla] = []

    for cla in partition[task]:
        other_class = partition[task][:]
        other_class.remove(cla)
        other = []
        for clas in other_class:
            other += train_per_class[clas]
        for idx in range(len(train_per_class[cla])):
            if type == 'feature':
                dist = pow(features[train_per_class[cla][idx]]-features[other],2)
            elif type == 'embedding':
                dist = pow(embeds[train_per_class[cla][idx]]-embeds[other],2)
            counts = np.sum(dist.cpu().detach().numpy(),1)
            cm[cla].append([train_per_class[cla][idx], len(counts[counts<distance])])
    
    for i in cm.keys():
        centrality = np.array([cm[i][ind][1] for ind in range(len(cm[i]))])
        proportion = min(int(len(train_per_class[i]) / total_data * size), len(train_per_class[i]))
        ind = np.argpartition(centrality, -proportion)[-proportion:]
        memory = [cm[i][idx][0] for idx in ind]
        memory = torch.from_numpy(np.array(memory))
        if replay_buffer == None:
            replay_buffer = memory
        else:
            replay_buffer = torch.cat((replay_buffer, memory),0)
    
    return replay_buffer, cm


# 根据策略类型（如 'random', 'MFf', 'CMe'）调用对应函数。
# 首次任务初始化字典（如 mean_features, cm），后续任务复用。
# 返回当前回放缓冲区及中间状态（均值、距离等）。
def update_replay(replay_type, network, edge_index, size, replay_buffer, total_data, partition, task, labels, train, features, device, train_per_class, distance, mean_features, count, dist, cm, replay, homophily, degree):
    with torch.no_grad():
        if task == len(train)-1:
            pass
        else:
            if replay_type == 'random':
                replay_buffer = None
                replay_buffer = update_random(train_per_class, partition, task, size, replay_buffer, total_data)
            elif replay_type == 'MFf':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, edge_index, 'feature', size, replay_buffer, total_data, partition, task, labels, train, train_per_class, features, device, mean_features, count, dist)
            elif replay_type == 'MFe':
                if task == 0:
                    mean_features, count, dist = {}, {}, {}
                replay_buffer = None
                replay_buffer, mean_features, count, dist = update_MF(network, edge_index, 'embedding', size, replay_buffer, total_data, partition, task, labels, train, train_per_class, features, device, mean_features, count, dist)
            elif replay_type == 'CMf':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, edge_index, 'feature', train_per_class, size, replay_buffer, total_data, partition, task, features, cm, distance)
            elif replay_type == 'CMe':
                if task == 0:
                    cm = {}
                replay_buffer = None
                replay_buffer, cm = update_CM(network, edge_index, 'embedding', train_per_class, size, replay_buffer, total_data, partition, task, features, cm, distance)
            elif replay_type =='CD':
                if task == 0:
                    replay = {}
                replay_buffer, replay = coverage_max(network, edge_index, train_per_class, size, replay, total_data, partition, task, features, distance)


    return replay_buffer, mean_features, count, dist, cm, replay, homophily, degree