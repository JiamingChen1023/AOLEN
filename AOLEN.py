import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats.distributions import chi2
import torch.nn.functional as F
import copy
import pdb
import gc
import torch.nn.init as init
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


class DynamicNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_size, hidden_size),
                                         nn.Linear(hidden_size, output_size)))
        self.layers.append(nn.Sequential(nn.Linear(hidden_size + fea, hidden_size), nn.ReLU(),
                                         nn.Linear(hidden_size, output_size)))
        self.num_layers = 2
        self.optimizer = []
        self.optimizer1 = []
        self.optimizer2 = []
        self.losses = []
        self.optzhu = []
        self.histloss = [0 for bb in range(self.num_layers)]
        self.histsample = [0 for bb in range(self.num_layers)]

    def opt(self, lr):
        for i in range(self.num_layers):
            self.optzhu = self.optzhu + list(self.layers[i][0].parameters(True))
            self.optimizer.append(optim.SGD(self.optzhu, lr=lr))
            self.optimizer1.append(optim.SGD(self.layers[i].parameters(), lr=lr))
            self.optimizer2.append(optim.SGD(self.layers[i][-1].parameters(), lr=lr))

    def add_layer(self, lr):
        hidden = self.layers[-1][0].out_features
        self.layers.insert(self.num_layers, nn.Sequential(nn.Linear(hidden + fea, hidden), nn.ReLU(),
                                                          nn.Linear(hidden, output_size)))
        self.optzhu = self.optzhu + list(self.layers[-1][0].parameters())
        self.optimizer.append(optim.SGD(self.optzhu, lr=lr))
        self.optimizer1.append(optim.SGD(self.layers[-1].parameters(), lr=lr))
        self.optimizer2.append(optim.SGD(self.layers[-1][-1].parameters(), lr=lr))
        self.histloss.append(0)
        self.histsample.append(0)
        loss_history.append([])

    def model_grow_and_prune(self, lr):
        exploss = [self.histloss[a] / self.histsample[a] for a in range(model.num_layers)]
        cop = exploss.index(min(exploss))
        sorted_indices = [idx for idx, acc in sorted(enumerate(exploss), key=lambda x: x[1])]
        self.add_layer(lr)
        self.num_layers += 1
        print(f"Add new layer, sample：{m+1}, Optimal layer: {cop+1}")
        if sorted_indices[0] != 0:
            self.layers[-1][0].weight.data.copy_(self.layers[sorted_indices[0]][0].weight.data)
            self.layers[-1][-1].weight.data.copy_(self.layers[sorted_indices[0]][-1].weight.data)
            self.layers[-1][0].bias.data.copy_(self.layers[sorted_indices[0]][0].bias.data)
            self.layers[-1][-1].bias.data.copy_(self.layers[sorted_indices[0]][-1].bias.data)
        else:
            self.layers[-1][0].weight.data.copy_(self.layers[sorted_indices[1]][0].weight.data)
            self.layers[-1][-1].weight.data.copy_(self.layers[sorted_indices[1]][-1].weight.data)
            self.layers[-1][0].bias.data.copy_(self.layers[sorted_indices[1]][0].bias.data)
            self.layers[-1][-1].bias.data.copy_(self.layers[sorted_indices[1]][-1].bias.data)

    def model_remove(self, list):
        if len(self.layers) > 1:
            layers_to_remove = 1
            self.layers = nn.ModuleList(self.layers[:-1])
            if list:
                list.pop()
            self.num_layers -= layers_to_remove
            print(f'删除最后一层，剩余层数：{self.num_layers}', m)
            if len(self.optimizer) > 0:
                self.optimizer.pop()
            if len(self.optimizer1) > 0:
                self.optimizer1.pop()
            if len(self.optimizer2) > 0:
                self.optimizer2.pop()
        else:
            print("模型至少保留一层，不执行删除操作")

    def forward(self, x):
        out = []
        x1 = x
        for layer in self.layers:
            if layer == self.layers[0]:
                x = layer[0](x)
                h = layer[1](x)
            else:
                x = layer[0](torch.cat((x, mzzz(x1)), dim=0))
                x = layer[1](x)
                h = layer[2](x)
            out.append(h)
        return out

    def forward1(self, x):
        out = []
        x1 = x
        for layer in self.layers:
            if layer == self.layers[0]:
                x = layer[0](x)
                h = layer[1](x)
            else:
                x = layer[0](torch.cat((x, mzzz(x1)), dim=1))
                x = layer[1](x)
                h = layer[2](x)
            out.append(h)
        return out

    def back(self, x, y, weights):
        self.losses = []
        y = y.long()
        for i in range(len(self.optimizer)):
            outputs = self.forward(x)
            output = outputs[i].unsqueeze(0)
            _, predicteaa = torch.max(output, 1)
            loss = criterion(output, y)
            self.losses.append(loss)
            self.histloss[i] += loss.item()
            self.histsample[i] += 1
        for j in range(len(self.optimizer)):
            self.optimizer[j].zero_grad()
            self.optimizer2[j].zero_grad()
            if next(model.layers[j].parameters()).requires_grad != False:
                self.losses[j].backward()
                torch.nn.utils.clip_grad_value_(self.layers[j][-1].parameters(), clip_value=10000)
                params = [param for param in self.optimizer[j].param_groups[0]['params']]
                torch.nn.utils.clip_grad_value_(params, clip_value=10000)
                for a, param in enumerate(params):
                    if param.grad is not None:
                        with torch.no_grad():
                            param.data -= lr * param.grad.data * weights[j]
                self.optimizer2[j].step()

    def newback11(self, data, label, weights):
        label = label.squeeze(1)
        label = label.long()
        for x, y in zip(data, label):
            losss = []
            for i in range(len(self.optimizer)):
                outputs = self.forward(x)
                output = outputs[i]
                loss = criterion(output, y)
                losss.append(loss)
            for j in range(len(self.optimizer)):
                self.optimizer[j].zero_grad()
                self.optimizer2[j].zero_grad()
                if next(model.layers[j].parameters()).requires_grad != False:
                    losss[j].backward()
                    params = [param for param in self.optimizer[j].param_groups[0]['params']]
                    for a, param in enumerate(params):
                        if param.grad is not None:
                            with torch.no_grad():
                                param.data -= lr * param.grad.data * weights[j]
                    self.optimizer2[j].step()

    def newback(self, x, y, i):
        y = y.squeeze(1)
        y = y.long()
        for g in range(len(self.optimizer)):
            outputs = self.forward(x)
            output = outputs[g]
            loss = criterion(output, y)
        self.optimizer[i].zero_grad()
        loss.backward()
        self.optimizer[i].step()
        for g in range(len(self.optimizer)):
            outputs = self.forward(x)
            output = outputs[g]
            loss = criterion(output, y)
            self.optimizer2[i].zero_grad()
            loss.backward()
            self.optimizer2[i].step()

    def newbackpp(self, x, y, i):
        y = y.squeeze(1)
        y = y.long()
        for g in range(len(self.optimizer)):
            outputs = self.forward(x)
            output = outputs[g]
            if g == i:
                loss = criterion(output, y)
                self.optimizer2[g].zero_grad()
                loss.backward()
                self.optimizer2[g].step()

    def back1(self, x, y, a):
        y = y.long()
        self.losses = []
        for i in range(len(self.optimizer)):
            outputs = self.forward(x)
            output = outputs[i].unsqueeze(0)
            loss = criterion(output, y)
            self.losses.append(loss)
            self.histloss[a] += loss
            self.histsample[a] += 1
        self.optimizer[a].zero_grad()
        (self.losses[a] * weight[a]).backward()
        self.optimizer[a].step()

    def newback1(self, x, y):
        y = y.squeeze(1)
        y = y.long()
        for i in range(len(self.optimizer)):
            outputs = self.forward1(x)
            output = outputs[i]
            loss = criterion(output, y)
            self.optimizer2[i].zero_grad()
            loss.backward()
            self.optimizer2[i].step()

    def newback12(self, x, y, i):
        y = y.squeeze(1)
        y = y.long()
        outputs = self.forward1(x)
        output = outputs[i]
        loss = criterion(output, y)
        optimizer = optim.SGD(model.layers[i].parameters(), lr=lr)
        optimizer.zero_grad()
        (loss).backward()
        optimizer.step()

    def newhidtrain1(self, data, label):
        label = label.squeeze(1)
        label = label.long()
        outputs = self.forward1(data)
        output = outputs[-1]
        loss = criterion(output, label)
        self.optimizer1[-1].zero_grad()
        loss.backward()
        self.optimizer1[-1].step()

    def outtrain(self, x, y, i):
        y = y.squeeze(1)
        y = y.long()
        outputs = self.forward(x)
        output = outputs[i]
        loss = criterion(output, y)
        self.optimizer2[i].zero_grad()
        loss.backward()
        self.optimizer2[i].step()


def yichang(data, label, anomalyc, avg, x, y, m, output):
    x = x.unsqueeze(0)
    output = output.unsqueeze(0)
    avg.calcMeanStd(x)
    if m>0:
        m = torch.tensor([m])
        anomalyc.updateAnomaly(x, y, m, avg.mean, output)
    nHl = 1
    anomalyc.addDataToAnomaly(data, label, nHl)
    return anomalyc


def ensemble(out, weight):
    pre = [out1 * weight1 for out1, weight1 in zip(out, weight)]
    sum_pre = sum(pre)
    return sum_pre


def updata_weights(weights, losses, beta):
    lossnp = [loss.detach().numpy() for loss in losses]
    M = sum(lossnp)
    lossnp = [loss / (M) for loss in lossnp]
    min_loss = np.min(lossnp)
    max_loss = np.max(lossnp)
    range_of_loss = (max_loss - min_loss) + 0.000001
    if range_of_loss != 0:
        lossnp = [(loss - min_loss) / range_of_loss for loss in lossnp]
    beta1 = [beta ** loss for loss in lossnp]
    alpha = [a * w for a, w in zip(beta1, weights)]
    alpha = [max(0.2 / model.num_layers, a) for a in alpha]
    M = sum(alpha)
    alpha = [a / M for a in alpha]
    weights = alpha
    return weights


class meanStd(object):
    def __init__(self):
        self.mean = 0.0
        self.mean_old = 0.0
        self.std = 0.001
        self.count = 0.0
        self.minMean = 100.0
        self.minStd = 100.0
        self.M_old = 0.0
        self.M = 0.0
        self.S = 0.0
        self.S_old = 0.0
        self.zql = []

    def calcMeanStd(self, data, cnt=1):
        self.data = data
        self.zql.append(data)
        self.mean_old = copy.deepcopy(self.mean)
        self.M_old = self.count * self.mean_old
        self.M = self.M_old + data
        self.S_old = copy.deepcopy(self.S)
        if self.count > 0:
            self.S = self.S_old + ((self.count * data - self.M_old) ** 2) / (self.count * (self.count + cnt))
        self.count += cnt
        self.mean = self.mean_old + np.divide((data - self.mean_old), self.count)
        self.std = np.sqrt(self.S / self.count)
        if (self.std != self.std).any():
            print('There is NaN in meanStd')
            pdb.set_trace()

    def resetMinMeanStd(self):
        self.minMean = copy.deepcopy(self.mean)
        self.minStd = copy.deepcopy(self.std)

    def calcMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd = copy.deepcopy(self.std)


class anomalyData(object):
    def __init__(self, nInput):
        self.Lambda = 0.98
        self.StabilizationPeriod = 20
        self.indexStableExecution = nInput
        self.na = 10
        self.Threshold1 = chi2.ppf(0.99, df=nInput)
        self.Threshold2 = chi2.ppf(0.999, df=nInput)
        self.indexkAnomaly = 0
        self.invCov = torch.eye(nInput, nInput)
        self.center = torch.zeros(1, nInput)
        self.caCounter = 0
        self.anomalyData = torch.Tensor().float()
        self.anomalyLabel = torch.Tensor().long()
        self.anomalyIndices = torch.Tensor().long()
        self.ChangePoints = []

    def reset(self):
        self.indexkAnomaly = 0
        self.invCov = torch.eye(fea, fea)
        self.center = torch.zeros(1, fea)
        self.caCounter = 0
        self.ChangePoints = []
        self.anomalyIndices = torch.Tensor().long()
        self.anomalyData = torch.Tensor().float()
        self.anomalyLabel = torch.Tensor().long()

    def updateCenterCov(self, x):
        with torch.no_grad():
            default_Eff_Number = 200
            indexOfSample = np.min([self.indexkAnomaly, default_Eff_Number])
            temp1 = self.mahalDist(x)
            temp1 = temp1 + (self.indexkAnomaly - 1) / self.Lambda
            multiplier = ((self.indexkAnomaly) / ((self.indexkAnomaly - 1) * self.Lambda))
            invCov = (self.invCov - (torch.matmul(torch.matmul(self.invCov, (x - self.center).transpose(0, 1)),
                                                  torch.matmul((x - self.center), self.invCov)) / temp1))
            self.invCov = multiplier * invCov
            self.center = self.Lambda * self.center + (1.0 - self.Lambda) * x

    def updateAnomaly(self, x, y, indice, avgX, score, cnt=1):
        with torch.no_grad():
            self.indexkAnomaly += cnt
            if self.indexkAnomaly <= self.indexStableExecution:
                self.center = avgX
            elif self.indexkAnomaly > self.indexStableExecution:
                mahaldist = self.mahalDist(x)
                sortedScore, _ = torch.sort(F.softmax(score, dim=1), descending=True)
                sortedScore = sortedScore.squeeze(dim=0).tolist()
                decisionBoundary = sortedScore[0] / (sortedScore[0] + sortedScore[1])
                if self.indexkAnomaly > self.StabilizationPeriod:
                    if ((mahaldist > self.Threshold1 and mahaldist < self.Threshold2)
                            or decisionBoundary <= 0.55):
                        self.anomalyIndices = torch.cat((self.anomalyIndices, indice), 0)
                    else:
                        self.caCounter += cnt
                if (self.caCounter >= self.na):
                    self.ChangePoints.append(self.indexkAnomaly - self.caCounter)
                    self.caCounter = 0
                self.updateCenterCov(x)

    def addDataToAnomaly(self, data, label, nHl):
        anomalyData = torch.index_select(data, 0, self.anomalyIndices)
        anomalyLabel = torch.index_select(label, 0, self.anomalyIndices)
        self.anomalyData = torch.cat((self.anomalyData, anomalyData), 0)
        self.anomalyLabel = torch.cat((self.anomalyLabel, anomalyLabel), 0)
        self.anomalyIndices = torch.Tensor().long()
        if self.anomalyData.shape[0] > 5000.0 * nHl:
            newIndex = int(5000.0 * nHl - self.anomalyData.shape[0])
            self.anomalyData = self.anomalyData[newIndex:]
            self.anomalyLabel = self.anomalyLabel[newIndex:]

    def mahalDist(self, x):
        with torch.no_grad():
            mahaldist = torch.matmul(torch.matmul((x - self.center), self.invCov), (x - self.center).transpose(0, 1))
            self.mahaldist = mahaldist[0][0].tolist()
        return mahaldist


def mzzz(data):
    mean_vals = torch.mean(data, dim=0)
    std_vals = torch.std(data, dim=0)
    std = torch.clamp(std_vals, min=1e-8)
    standardized_data = (data - mean_vals) / std
    return standardized_data


def run_experiments(dataset, dataflow_settings, model_settings,randomfactor=None):
    global fea, output_size, model, criterion, loss_history, m, lr, weight

    TTime = dataflow_settings['TTime']
    k = dataflow_settings['k']
    beta = dataflow_settings['beta']

    hidden_size = model_settings['hidden_size']
    lr = model_settings['lr']

    start = time.perf_counter()
    if dataset == 'rialto' or dataset == 'MIRS':
        dataz = pd.read_csv("../../Datasets/{}.csv".format(dataset), skiprows=1, header=None)
        num = dataz.columns
        num = len(num)
        data = dataz.iloc[:, :num-1]
        label = dataz.iloc[:, num-1:num]-1
        dff = label.drop_duplicates(subset=[num-1], keep='first')
        dff = dff[num-1]
        print(dataset)
        labelnum = len(dff.tolist())
        example = (len(data.axes[0]))
        fea = (len(data.axes[1]))
    elif dataset == 'CANCER':
        dataz = pd.read_csv("../../Datasets/{}.csv".format(dataset), skiprows=1, header=None)
        num = dataz.columns
        num = len(num)
        data = dataz.iloc[:, :num-1]
        label = dataz.iloc[:, num-1:num]
        dff = label.drop_duplicates(subset=[num-1], keep='first')
        dff = dff[num-1]
        labelnum = len(dff.tolist())
        example = (len(data.axes[0]))
        fea = (len(data.axes[1]))
    else:
        data = pd.read_csv("../../Datasets/{}data.csv".format(dataset), header=None)
        label = pd.read_csv("../../Datasets/{}label.csv".format(dataset), header=None)
        dff = label.drop_duplicates(subset=0, keep='first')
        dff = dff[0]
        labelnum = len(dff.tolist())
        example = (len(data.axes[0]))
        fea = (len(data.axes[1]))
    correct = 0
    total = 0
    anomalyc = anomalyData(fea)
    data = data.to_numpy()
    label = label.to_numpy()
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    data = data.float()
    label = label.float()
    if randomfactor!=None:
        torch.manual_seed(randomfactor)
    input_size = fea
    output_size = labelnum
    model = DynamicNet(input_size, hidden_size, output_size)
    avg = meanStd()
    criterion = nn.CrossEntropyLoss()
    model.opt(lr)
    obsy = []
    obsx = []
    raccy = []
    raccx = []
    ti = 0
    m = 0
    b = 0
    c = 0
    d = 0
    allm = []
    alk = []
    y_t = []
    y_p = []
    weightold = [0.5, 0.5]
    loss_history = []
    weight = [1.0 / model.num_layers for _ in range(model.num_layers)]
    rise = []
    sk = k
    for h in range(int((example + TTime - 1) / TTime)):
        tot = 0
        cor = 0
        if h < int((example + TTime - 1) / TTime) - 1:
            current_batch_size = TTime
        else:
            current_batch_size = example - TTime * h
        for t in range(current_batch_size):
            inputs = data[m]
            labels = label[m]
            model.eval()
            outputs = model(inputs)
            softout = []
            for i in range(len(outputs)):
                probabilities = F.softmax(outputs[i], dim=0)
                softout.append(probabilities)
                output = ensemble(softout, weight)
            _, predicted = torch.max(output, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
            tot += labels.size(0)
            cor += (predicted == labels).sum().item()
            y_p.append(predicted)
            y_t.append(labels)
            obsy.append(accuracy)
            obsx.append(m + 1)
            model.back(inputs, labels, weight)
            losses = model.losses
            weight = updata_weights(weight, losses, beta=beta)
            anomalyc = yichang(data, label, anomalyc, avg, inputs, labels, m, output)
            if weight.index(max(weight)) + 1 == len(weight):
                b += 1
            if weight.index(max(weight)) + 1 != len(weight):
                b = 0
            if weightold[-1] <= weight[-1] and weight[-1] != 0.2 / model.num_layers:
                c += 1
                rise.append(1)
            if weightold[-1] > weight[-1]:
                d += 1
                rise.append(0)
            if len(rise) > 1000:
                rise = rise[1:]
            if b >= 1000 and sum(rise) >= k:
                weight.append(max(weight))
                sk = sum(rise) + sk
                k = sk / model.num_layers
                M = sum(weight)
                allm.append(m)
                alk.append(k)
                weight = [a / M for a in weight]
                model.model_grow_and_prune(lr)
                model.histloss = [0 for bb in range(model.num_layers)]
                model.histsample = [0 for bb in range(model.num_layers)]
                model.newback11(anomalyc.anomalyData, anomalyc.anomalyLabel, weight)
                print(f'k:{k}')
                b = 0
                c = 0
                d = 0
            weightold = weight
            m += 1
        ti += 1
        raccy.append(cor / tot)
        raccx.append(ti)
        if m % 10000 == 0:
            print(f"Sample: {m}, Weight: {weight}, Cumulative Accuracy:{accuracy}")
    F1 = f1_score(y_t, y_p, average='macro')
    mcc = matthews_corrcoef(y_t, y_p)
    cm = confusion_matrix(y_t, y_p)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    size = get_model_size(model)
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Dataset: {m}, Number of layers : {model.num_layers}, Model size: {size:.2f} MB, time: {elapsed_time} s")
    print(f"Cumulative Accuracy:{accuracy}，F1:{F1}, MCC:{mcc}")