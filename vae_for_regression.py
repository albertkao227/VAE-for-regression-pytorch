import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import MDS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score 


def sampling(mean, log_var):
    '''
    Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    Returns:
        z (tensor): sampled latent vector
    '''
    #print(type(mean), type(log_var))
    epsilon = torch.randn_like(mean) 
    return mean + torch.exp(0.5*log_var)*epsilon 


def augment_by_transformation(data,age,n):
    augment_scale = 1
    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        for i in range(0,m):
            new_data = np.zeros((1,data.shape[1],data.shape[2],data.shape[3],1))
            idx = np.random.randint(0,raw_n)
            new_age = age[idx]
            new_data[0] = data[idx].copy()
            new_data[0,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[0,:,:,:,0],np.random.uniform(-1,1),axes=(1,0),reshape=False)
            new_data[0,:,:,:,0] = sp.ndimage.interpolation.rotate(new_data[0,:,:,:,0],np.random.uniform(-1,1),axes=(0,1),reshape=False)
            new_data[0,:,:,:,0] = sp.ndimage.shift(new_data[0,:,:,:,0],np.random.uniform(-1,1))
            data = np.concatenate((data, new_data), axis=0)
            age = np.append(age, new_age)
        
        return data,age
    

class Encoder(nn.Module):
    def __init__(self, input_shape_x, intermediate_dim, latent_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(input_shape_x, 128)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(128, intermediate_dim)
        self.act2 = nn.Tanh() 
        
        # posterior on Y; probabilistic regressor 
        self.r_mean_layer = nn.Linear(intermediate_dim, 1)
        self.r_logvar_layer = nn.Linear(intermediate_dim, 1) 

        # q(z|x) 
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_logvar_layer = nn.Linear(intermediate_dim, latent_dim)

        # latent generator 
        self.gen_z = weight_norm(nn.Linear(1, latent_dim))

    def forward(self, x):
        x = self.dropout(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))

        r_mean = self.r_mean_layer(x)
        r_logvar = self.r_logvar_layer(x)

        z_mean = self.z_mean_layer(x)
        z_logvar = self.z_logvar_layer(x)
                
        # reparameterization trick
        r = sampling(r_mean, r_logvar)
        z = sampling(z_mean, z_logvar)

        pz_mean = self.gen_z(r) 

        return r_mean, r_logvar, r, z_mean, z_logvar, z, pz_mean


class Decoder(nn.Module):
    def __init__(self, input_shape_x, intermediate_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(intermediate_dim, 128)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(128, input_shape_x)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x 


####### Main Script #######
min_x = int(sys.argv[1])    
min_y = int(sys.argv[2])  
min_z = int(sys.argv[3])  
patch_x = int(sys.argv[4])    
patch_y = int(sys.argv[5])    
patch_z = int(sys.argv[6])   

dropout_alpha = float(sys.argv[7])     
L2_reg = float(sys.argv[8]) 


## CNN Parameters 
#dropout_alpha = 0.5
ft_bank_baseline = 16
latent_dim = 16
augment_size = 1000
#L2_reg= 0.00
binary_image = False


## load data
file_idx = np.loadtxt('./access.txt')  
age = np.loadtxt('./age.txt') 
subject_num = file_idx.shape[0]


## Cross Validation 
print("Data size \n",data.shape)
skf = StratifiedKFold(n_splits=5,shuffle=True)
fake = np.zeros((data.shape[0]))
pred = np.zeros((age.shape))


for train_index, test_index in skf.split(data, fake): 

    train_data = data[train_idx]
    train_age = age[train_idx]
    test_data = data[test_idx]
    test_age = age[test_idx]

    