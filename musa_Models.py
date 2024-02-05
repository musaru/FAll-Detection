import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.layers import *
#from utils import import_class
#from Utils import Graph
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
def count_params(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'fc' not in name)
def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'swish':
        return Swish()
    elif name == 'hardswish':
        return HardSwish()
    elif name == 'metaacon':
        return MetaAconC()
    elif name == 'acon':
        return AconC()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)
class Randomized_DropBlock_Ske(nn.Module):
    def __init__(self, block_size=7):
        super(Randomized_DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob, A, num_point):  # n,c,t,v
        self.keep_prob = keep_prob
        self.num_point = num_point
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        #print(input.shape) 
        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2 
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            warnings.warn('undefined skeleton graph')
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        #print(M_seed.shape)
        print(A.shape)
        M = torch.matmul(M_seed, A)
        #M = torch.einsum('nv,cvw->nv', (M_seed, A)).contiguous()
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        #print(M.shape)
        mask = (1 - M).view(n, 1, 1, self.num_point)
        #print(M.shape)
        return input * mask * mask.numel() / mask.sum()
    
    
class Randomized_DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(Randomized_DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        idx = torch.randperm(Msum.shape[2])
        RMsum = Msum[:,:,idx].view(Msum.size()) ## shuffles MSum to drop random frames instead of dropping a block of frames
        mask = (1 - RMsum).to(device=input.device, dtype=input.dtype)
        #print(mask.shape)
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, act_type, keep_prob, block_size, 
                 num_point, residual=True, **kwargs):
        super(SpatialGraphConv, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
            
        self.act = activation_factory(act_type)   
        self.bn = nn.BatchNorm2d(out_channel)
        
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
         
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        n, kc, t, v = x.size()
        #x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        print((self.A*self.edge).shape)#torch.Size([3, 14, 14])
        print(x.shape)#torch.Size([32, 128, 29, 14])
        x = torch.einsum('nctv,cvw->nctw', (x, self.A * self.edge)).contiguous()#n=バッチサイズ,1
        print(self.A * self.edge)
        #x = self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point) + self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point)
        x = self.dropT(self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        
        return self.act(x)
    
class SepTemporal_Block(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act_type, edge, A, num_point, keep_prob, block_size, expand_ratio, stride=1, residual=True, **kwargs):
        super(SepTemporal_Block, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        padding = (temporal_window_size - 1) // 2
        self.act = activation_factory(act_type)

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = lambda x:0
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
        
    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        #x = self.dropT(x, self.keep_prob) + self.dropT(res, self.keep_prob)
        x = self.dropT(self.dropS(x, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        return self.act(x)
class adjGraph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='ntu-rgb+d',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        elif layout == 'coco_cut':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                             (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)]
            self.edge = self_link + neighbor_link
            self.center = 13
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        #self.bn = nn.BatchNorm1d(dim* 25)#dim=2
        self.bn = nn.BatchNorm1d(dim* 14)#dim=2

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x
    
class embed(nn.Module):
    def __init__(self, dim, dim1, att_type, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(                        
                norm_data(dim),
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        #self.attention =  Attention_Layer(dim1,  att_type=att_type)

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        return x#self.attention(x)


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 max_frame,
                 graph,
                 act_type, 
                 bias,
                 edge,
                 block_size):
        super(Model, self).__init__()
        
        self.num_class =  num_class
        temporal_window_size = 3
        max_graph_distance = 2
        keep_prob = 0.9
        #Graph = import_class(graph)
        Graph = graph
        #A_binary = torch.Tensor(Graph().A_binary)
        A_binary = torch.Tensor(Graph.A)
        #A = torch.rand(3,25,25).cuda()#.to(num_class.dtype).to(num_class.device)
        #self.graph_hop = adjGraph(**graph_args)
        #A = torch.tensor(self.graph_hop.A, dtype=torch.float32, requires_grad=False)
        #self.register_buffer('A', A)
        
        # channels
        D_embed = 64
        c1 = D_embed*2
        c2 = c1 * 2     
        #c3 = c2 * 2    
       
        
        
        self.joint_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias)
        #self.dif_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias) #601
        #self.attention =  Attention_Layer(D_embed,  max_frame, act_type, att_type='stja')
        
        self.sgcn1 = SpatialGraphConv(D_embed, c1, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True)
        self.tcn11 = SepTemporal_Block(c1, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True)
        self.tcn12 = SepTemporal_Block(c1, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        
        self.sgcn2 = SpatialGraphConv(c1, c2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True)
        self.tcn21 = SepTemporal_Block(c2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True)
        self.tcn22 = SepTemporal_Block(c2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        
        
        self.fc = nn.Linear(c2, num_class)
        #init_param(self.modules())
    
    def forward(self, x):        
        #print(x.shape()
        #N, C, T, V = x.size()
        N, C, T, V = x[1].size()#42必要
        #dy = x
        
        # Dynamic Representation        
        pos = x[1].permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        #print(pos.shape):torch.Size([1, 2, 25, 300])
        #dif = pos[:, :, :, 1:] - pos[:, :, :, 0:-1] #  
        #dif = torch.cat([dif.new(N, dif.size(1), V, 1).zero_(), dif], dim=-1)
        
        pos = self.joint_embed(pos)        
        #dif = self.dif_embed(dif)
        dy = pos #+ dif
        #dy = dif
        dy = dy.permute(0,1,3,2).contiguous() # N, C, T, V   
        #print(dy.shape):torch.Size([1, 64, 300, 25])
        #dy = self.attention(dy)
        #dy.register_hook(lambda g: print(g))
      
        #########################
        out = self.sgcn1(dy)
        print(out.size())#torch.Size([1, 128, 300, 25])
        out = self.tcn11(out)
        print(out.size())#torch.Size([1, 128, 300, 25])
        out = self.tcn12(out)
        print(out.size())#torch.Size([1, 128, 150, 25])
        #out = self.tcn12(self.tcn11(self.sgcn1(dy)))
        out = self.sgcn2(out)
        print(out.size())#torch.Size([1, 256, 150, 25])
        out = self.tcn21(out)
        print(out.size())#torch.Size([1, 256, 150, 25])
        out = self.tcn22(out)
        print(out.size())#torch.Size([1, 256, 75, 25])
        #out = self.tcn22(self.tcn21(self.sgcn2(out)))
        #print(out.shape)
        out_channels = out.size(1)
        out = out.reshape(N, out_channels, -1)   
        print(out.shape)#torch.Size([1, 256, 1875])
        out = out.mean(2)
        print(out.shape)#torch.Size([1, 256])
        out = self.fc(out)
        print(out.size())#torch.Size([1, 2])
        return out