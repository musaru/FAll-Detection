import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from tqdm import tqdm
from torch.utils import data
from torch.optim.adadelta import Adadelta
from sklearn.model_selection import train_test_split
from musa_Models import Model,adjGraph

#二分割できる（訓練用とテスト用）

#from pose_utils import motions_map
import sys
sys.path.append("../")
from Actionsrecognition.Models import *
#from visualizer import plot_graphs, plot_confusion_metrix
#from sklearn.metrics import plot_confusion_metrix

save_folder = 'saved/TSSTG(pts+mot)-01(cf+hm-hm)'

#device = 'cuda'
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using", device, "device")

epochs = 30
batch_size = 32 #32

# DATA FILES.
# Should be in format of
#  inputs: (N_samples, time_steps, graph_node, channels),
#  labels: (N_samples, num_class)
#   and do some of normalizations on it. Default data create from:
#       Data.create_dataset_(1-3).py
# where
#   time_steps: Number of frame input sequence, Default: 30
#   graph_node: Number of node in skeleton, Default: 14
#   channels: Inputs data (x, y and scores), Default: 3
#   num_class: Number of pose class to train, Default: 7
'''
data_files = ['../Data/2clCoffee_01_new-set(labelXscrw).pkl',
              '../Data/2clHome_01_new-set(labelXscrw).pkl',
              '../Data/2clHome_02_new-set(labelXscrw).pkl']
'''

data_files = ['../Data/2clCoffee_01_new-set(labelXscrw).pkl',
              '../Data/2clHome_01_new-set(labelXscrw).pkl']

#data_files = ['../Data/ur2cl_new-set(labelXscrw).pkl']
#'../Data/2clHome_02_new-set(labelXscrw).pkl'
#class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               #'Stand up', 'Sit down', 'Fall Down']
class_names = ['Fall','No_fall']
num_class = len(class_names)


def load_dataset(data_files, batch_size, split_size=0.2):#0.2
    """Load data files into torch DataLoader with/without spliting train-test.
    """
    features, labels = [], []
    for fil in data_files:
        with open(fil, 'rb') as f:
            fts, lbs = pickle.load(f)
            features.append(fts)
            labels.append(lbs)
        del fts, lbs
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    if split_size > 0:
        x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=split_size,
                                                              random_state=9)
        train_set = data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_train, dtype=torch.float32))
        valid_set = data.TensorDataset(torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(y_valid, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_set, batch_size)
    else:
        train_set = data.TensorDataset(torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                                       torch.tensor(labels, dtype=torch.float32))
        train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = None
    return train_loader, valid_loader


def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).mean()


def set_training(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
    model.train(mode)
    return model


if __name__ == '__main__':
    save_folder = os.path.join(os.path.dirname(__file__), save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # DATA.
    train_loader, _ = load_dataset(data_files[0:1], batch_size) #batch_size = 32
    
    valid_loader, train_loader_ = load_dataset(data_files[:1], batch_size, 0.2)
    #print("err")
    train_loader = data.DataLoader(data.ConcatDataset([train_loader.dataset, train_loader_.dataset]),
                                   batch_size, shuffle=True)
    dataloader = {'train': train_loader, 'valid': valid_loader}
    del train_loader_
    
    #print(train_loader.shape)
    
    # MODEL.(list化)
    model = Model(
        num_class=2,
        num_point=25,
        max_frame=300,
        #graph='graph.ntu_rgb_d.AdjMatrixGraph',
        graph=adjGraph(layout='coco_cut',
                      strategy='spatial'),
        act_type = 'relu',
        bias = True,
        edge = True,
        block_size=41
    ).to(device)
    graph_args = {'strategy': 'spatial'}
    graph_args = {'strategy': 'uniform'}
    #model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)
    #model = TwoStreamSpatialTemporalGraph(graph_args, 8).to(device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    #optimizer = Adadelta(model.parameters())
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    
    
    losser = torch.nn.BCELoss() #fall or no_fall
    #losser = torch.nn.CrossEntropyLoss()

    # TRAINING.
    loss_list = {'train': [], 'valid': []}
    accu_list = {'train': [], 'valid': []}
    best_acc = -1
    for e in range(epochs):
        print('Epoch {}/{}'.format(e, epochs - 1))
        for phase in ['train', 'valid']:
            if phase == 'train':
                model = set_training(model, True)
            else:
                model = set_training(model, False)

            run_loss = 0.0
            run_accu = 0.0
            with tqdm(dataloader[phase], desc=phase) as iterator:
                for pts, lbs in iterator:
                    # Create motion input by distance of points (x, y) of the same node
                    # in two frames.
                    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
                    
                    mot = mot.to(device)
                    pts = pts.to(device)
                    lbs = lbs.to(device)
                    #print(pts.size())torch.Size([32, 3, 30, 14])
                    #print(mot.size())torch.Size([32, 2, 29, 14])
                    # Forward.
                    out = model((pts, mot))#タプル型
                    #print(lbs)

                    #print(out)
                    loss = losser(out, lbs)

                    if phase == 'train':
                        # Backward.
                        model.zero_grad()
                        loss.backward()
                        optimizer.step()

                    run_loss += loss.item()
                    accu = accuracy_batch(out.detach().cpu().numpy(),
                                          lbs.detach().cpu().numpy())
                    run_accu += accu

                    iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                        loss.item(), accu))
                    iterator.update()
                    #break
            loss_list[phase].append(run_loss / len(iterator))
            accu_list[phase].append(run_accu / len(iterator))
            #print(accu_list)
            #print(torch.max(accu_list))
        if(best_acc < accu_list['valid'][-1]):
            best_acc = accu_list['valid'][-1]
            torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model_best.pth'))
            #break

        print('Summary epoch:\n - Train loss: {:.4f}, accu: {:.4f}\n - Valid loss:'
              ' {:.4f}, accu: {:.4f}'.format(loss_list['train'][-1], accu_list['train'][-1],
                                             loss_list['valid'][-1], accu_list['valid'][-1]))

        # SAVE.
        '''
        if(best_acc < accu_list['valid'][-1]):
            best_acc = accu_list['valid'][-1]
            torch.save(model.state_dict(), os.path.join(save_folder, 'tsstg-model_best.pth'))
            '''
        '''
        plot_graphs(list(loss_list.values()), list(loss_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            loss_list['train'][-1], loss_list['valid'][-1]
                        ), 'Loss', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'loss_graph.png'))
        plot_graphs(list(accu_list.values()), list(accu_list.keys()),
                        'Last Train: {:.2f}, Valid: {:.2f}'.format(
                            accu_list['train'][-1], accu_list['valid'][-1]
                        ), 'Accu', xlim=[0, epochs],
                        save=os.path.join(save_folder, 'accu_graph.png'))
        '''
            #break

    del train_loader, valid_loader

    #model.load_state_dict(torch.load(os.path.join(save_folder, 'tsstg-model.pth',map_location=torch.device('cpu'))))
    model.load_state_dict(torch.load(os.path.join(save_folder, 'tsstg-model_best.pth')))
    # EVALUATION.
    model = set_training(model, False)
    data_file = data_files[1]
    eval_loader, _ = load_dataset([data_file], 32)
    #data_file = data_files[2]
    #eval_loader, _ = load_dataset([data_file], 64)

    print('Evaluation.')
    run_loss = 0.0
    run_accu = 0.0
    y_preds = []
    y_trues = []
    #with tqdm(eval_loader, desc='eval') as iterator:
    #URFD
    with tqdm(dataloader[phase], desc='eval') as iterator:
        for pts, lbs in iterator:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot = mot.to(device)
            pts = pts.to(device)
            lbs = lbs.to(device)

            out = model((pts, mot))
            loss = losser(out, lbs)

            run_loss += loss.item()
            accu = accuracy_batch(out.detach().cpu().numpy(),
                                  lbs.detach().cpu().numpy())
            run_accu += accu

            y_preds.extend(out.argmax(1).detach().cpu().numpy())
            y_trues.extend(lbs.argmax(1).cpu().numpy())

            iterator.set_postfix_str(' loss: {:.4f}, accu: {:.4f}'.format(
                loss.item(), accu))
            iterator.update()

    run_loss = run_loss / len(iterator)
    run_accu = run_accu / len(iterator)

    '''plot_confusion_metrix(y_trues, y_preds, class_names, 'Eval on: {}\nLoss: {:.4f}, Accu{:.4f}'.format(
        os.path.basename(data_file), run_loss, run_accu
    ), 'true', save=os.path.join(save_folder, '{}-confusion_matrix.png'.format(
        os.path.basename(data_file).split('.')[0])))
        '''
    print('Eval Loss: {:.4f}, Accu: {:.4f}'.format(run_loss, run_accu))

