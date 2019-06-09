import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')





#########################  AE ###############
# class AE(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.latent_size = latent_size

#         self.encoder = Encoder(latent_size)
#         self.decoder = Decoder(latent_size)

#     def forward(self, x):
        
#         x = self.encoder(x)
# #         x = x.unsqueeze_(2)
#         x = self.decoder(x)

#         return x


# ###############3 VAE_CVAE (github:https://github.com/timbmg/VAE-CVAE-MNIST) #####################
# class VAE(nn.Module):

#     def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):

#         super().__init__()

#         assert type(encoder_layer_sizes) == list
#         assert type(latent_size) == int
#         assert type(decoder_layer_sizes) == list

#         self.latent_size = latent_size

#         self.encoder = Encoder(
#             encoder_layer_sizes, latent_size)
#         self.decoder = Decoder(
#             decoder_layer_sizes, latent_size)

#     def forward(self, x, c=None):

#         if x.dim() > 2:
#             x = x.view(-1, 28*28)

#         batch_size = x.size(0)

#         means, log_var = self.encoder(x, c)

#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn([batch_size, self.latent_size])
#         z = eps * std + means

#         recon_x = self.decoder(z, c)

#         return recon_x, means, log_var, z

#     def inference(self, n=1, c=None):

#         batch_size = n
#         z = torch.randn([batch_size, self.latent_size])

#         recon_x = self.decoder(z, c)

#         return recon_x


# class Encoder(nn.Module):

#     def __init__(self, layer_sizes, latent_size):

#         super().__init__()

#         self.MLP = nn.Sequential()

#         for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
#             self.MLP.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

#         self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
#         self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

#     def forward(self, x, c=None):


#         x = self.MLP(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, layer_sizes, latent_size):

#         super().__init__()

#         self.MLP = nn.Sequential()

#         input_size = latent_size

#         for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
#             self.MLP.add_module(
#                 name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             if i+1 < len(layer_sizes):
#                 self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
#             else:
#                 self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

#     def forward(self, z, c):

#         x = self.MLP(z)

#         return x


#########################  VAE ####################
class VAE(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.latent_size = latent_size

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        
        batch_size = x.size(0)

        means, log_var = self.encoder(x)

        # reparameterization trick
        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_size]).to(device)
        mul = torch.mul(eps, std).to(device)
        z = torch.add(mul,means).to(device)

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def inference(self, n=1):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size]).to(device)

        recon_x = self.decoder(z)

        return recon_x

    ########## 1002*11, 5000, 1000, 500, 256,(All_Sides), with batch normalizetion for VAE and sigmoid end, Leaky_relu #########
class Encoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()


        self.fc1 = nn.Linear(1002*11,5000)
        self.fc2 = nn.Linear(5000,1000)
        self.fc3 = nn.Linear(1000,500)
        self.fc4 = nn.Linear(500,256)



        self.linear_means = nn.Linear(256, latent_size)
        self.linear_log_var = nn.Linear(256, latent_size)
        
        self.fc_bn1 = nn.BatchNorm1d(5000)
        self.fc_bn2 = nn.BatchNorm1d(1000)
        self.fc_bn3 = nn.BatchNorm1d(500)
        self.fc_bn4 = nn.BatchNorm1d(256)


        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn4(x)
        x = self.dropout(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_size):

        super().__init__()

        self.fc5 = nn.Linear(latent_size,256)
        self.fc6 = nn.Linear(256, 500)
        self.fc7 = nn.Linear(500, 1000)
        self.fc8 = nn.Linear(1000, 5000)
        self.fc9 = nn.Linear(5000, 1002 * 11)

        
        self.fc_bn5 = nn.BatchNorm1d(256)
        self.fc_bn6 = nn.BatchNorm1d(500)
        self.fc_bn7 = nn.BatchNorm1d(1000)
        self.fc_bn8 = nn.BatchNorm1d(5000)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        x = self.fc5(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn5(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn6(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn7(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.fc_bn8(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc9(x))

        return x 
    

# ########## 1002*10, 5000, 1000, 500, 256,(All_Sides), with batch normalizetion for VAE and sigmoid end, Leaky_relu #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(1002*10,5000)
#         self.fc2 = nn.Linear(5000,1000)
#         self.fc3 = nn.Linear(1000,500)
#         self.fc4 = nn.Linear(500,256)



#         self.linear_means = nn.Linear(256, latent_size)
#         self.linear_log_var = nn.Linear(256, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(5000)
#         self.fc_bn2 = nn.BatchNorm1d(1000)
#         self.fc_bn3 = nn.BatchNorm1d(500)
#         self.fc_bn4 = nn.BatchNorm1d(256)


#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = self.fc1(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn1(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn2(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn3(x)
#         x = self.dropout(x)
#         x = self.fc4(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn4(x)
#         x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc5 = nn.Linear(latent_size,256)
#         self.fc6 = nn.Linear(256, 500)
#         self.fc7 = nn.Linear(500, 1000)
#         self.fc8 = nn.Linear(1000, 5000)
#         self.fc9 = nn.Linear(5000, 1002 * 10)

        
#         self.fc_bn5 = nn.BatchNorm1d(256)
#         self.fc_bn6 = nn.BatchNorm1d(500)
#         self.fc_bn7 = nn.BatchNorm1d(1000)
#         self.fc_bn8 = nn.BatchNorm1d(5000)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = self.fc5(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn5(x)
#         x = self.dropout(x)
#         x = self.fc6(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn6(x)
#         x = self.dropout(x)
#         x = self.fc7(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn7(x)
#         x = self.dropout(x)
#         x = self.fc8(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn8(x)
#         x = self.dropout(x)
#         x = torch.sigmoid(self.fc9(x))

#         return x 
    
    
# ######################(without Counter, Time) 12,32,64,8 with batch normalizetion for VAE on Dropout#########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(10,32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,8)
#         self.linear_means = nn.Linear(8, latent_size)
#         self.linear_log_var = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(64)
#         self.fc_bn3 = nn.BatchNorm1d(8)

#         # self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         # x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,64)
#         self.fc6 = nn.Linear(64,32)
#         self.fc7 = nn.Linear(32,10)
        
#         self.fc_bn4 = nn.BatchNorm1d(8)
#         self.fc_bn5 = nn.BatchNorm1d(64)
#         self.fc_bn6 = nn.BatchNorm1d(32)

#         # self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn5(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc6(x))
#         x = self.fc_bn6(x)
#         # x = self.dropout(x)
#         x = torch.sigmoid(self.fc7(x))

#         return x

    
    
# ######################### 12,32,64,8 with batch normalizetion for VAE on Dropout#########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,8)
#         self.linear_means = nn.Linear(8, latent_size)
#         self.linear_log_var = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(64)
#         self.fc_bn3 = nn.BatchNorm1d(8)

#         # self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         # x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,64)
#         self.fc6 = nn.Linear(64,32)
#         self.fc7 = nn.Linear(32,12)
        
#         self.fc_bn4 = nn.BatchNorm1d(8)
#         self.fc_bn5 = nn.BatchNorm1d(64)
#         self.fc_bn6 = nn.BatchNorm1d(32)

#         # self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn5(x)
#         # x = self.dropout(x)
#         x = F.relu(self.fc6(x))
#         x = self.fc_bn6(x)
#         # x = self.dropout(x)
#         x = torch.sigmoid(self.fc7(x))

#         return x

# ######################### 784, 256,(MNIST), with batch normalizetion for VAE and sigmoid end, Leaky_relu #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(28 * 28,256)

#         self.linear_means = nn.Linear(256, latent_size)
#         self.linear_log_var = nn.Linear(256, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(256)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = self.fc1(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn1(x)
#         x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc2 = nn.Linear(latent_size,256)
#         self.fc3 = nn.Linear(256,28 * 28)
        
#         self.fc_bn2 = nn.BatchNorm1d(256)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = self.fc2(x)
#         x = F.leaky_relu(x, negative_slope=0.0001)
#         x = self.fc_bn2(x)
#         x = self.dropout(x)
#         x = torch.sigmoid(self.fc3(x))

#         return    
    
# ######################### 784, 256,(MNIST)   with batch normalizetion for VAE and sigmoid end #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(28 * 28,256)

#         self.linear_means = nn.Linear(256, latent_size)
#         self.linear_log_var = nn.Linear(256, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(256)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc2 = nn.Linear(latent_size,256)
#         self.fc3 = nn.Linear(256,28 * 28)
        
#         self.fc_bn2 = nn.BatchNorm1d(256)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = self.dropout(x)
#         x = F.sigmoid(self.fc3(x))

#         return x

    
    
# ######################### 784,1000,500,250  with batch normalizetion for VAE #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(28 * 28,1000)
#         self.fc2 = nn.Linear(1000, 500)
#         self.fc3 = nn.Linear(500, 250)
#         self.linear_means = nn.Linear(250, latent_size)
#         self.linear_log_var = nn.Linear(250, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(1000)
#         self.fc_bn2 = nn.BatchNorm1d(500)
#         self.fc_bn3 = nn.BatchNorm1d(250)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         x = self.dropout(x)

#         means = self.linear_means(x)
#         log_vars = self.linear_log_var(x)

#         return means, log_vars


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,250)
#         self.fc5 = nn.Linear(250,500)
#         self.fc6 = nn.Linear(500,1000)
#         self.fc7 = nn.Linear(1000,28 * 28)
        
#         self.fc_bn4 = nn.BatchNorm1d(250)
#         self.fc_bn5 = nn.BatchNorm1d(500)
#         self.fc_bn6 = nn.BatchNorm1d(1000)

#         self.dropout = nn.Dropout(0.5)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn5(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc6(x))
#         x = self.fc_bn6(x)
#         x = self.dropout(x)
#         x = F.relu(self.fc7(x))

#         return x


############################### 12,784,1000,500,250  Hinton's reduction paper #########

# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,784)
#         self.fc2 = nn.Linear(784,1000)
#         self.fc3 = nn.Linear(1000,500)
#         self.fc4 = nn.Linear(500,250)
#         self.fc5 = nn.Linear(250, latent_size)

#         self.fc_bn1 = nn.BatchNorm1d(784)
#         self.fc_bn2 = nn.BatchNorm1d(1000)
#         self.fc_bn3 = nn.BatchNorm1d(500)
#         self.fc_bn4 = nn.BatchNorm1d(250)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         x = F.relu(self.fc5(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc6 = nn.Linear(latent_size,250)
#         self.fc7 = nn.Linear(250,500)
#         self.fc8 = nn.Linear(500,1000)
#         self.fc9 = nn.Linear(1000,784)
#         self.fc10 = nn.Linear(784,12)

#         self.fc_bn5 = nn.BatchNorm1d(250)
#         self.fc_bn6 = nn.BatchNorm1d(500)
#         self.fc_bn7 = nn.BatchNorm1d(1000)
#         self.fc_bn8 = nn.BatchNorm1d(784)


#     def forward(self, x):

#         x = F.relu(self.fc6(x))
#         x = self.fc_bn5(x)
#         x = F.relu(self.fc7(x))
#         x = self.fc_bn6(x)
#         x = F.relu(self.fc8(x))
#         x = self.fc_bn7(x)
#         x = F.relu(self.fc9(x))
#         x = self.fc_bn8(x)
#         x = F.relu(self.fc10(x))

#         return x


# ############################### 12,32,64,8 with batch normalizetion #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,64)
#         self.fc3 = nn.Linear(64,8)
#         self.fc4 = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(64)
#         self.fc_bn3 = nn.BatchNorm1d(8)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = F.relu(self.fc3(x))
#         x = self.fc_bn3(x)
#         x = F.relu(self.fc4(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,64)
#         self.fc6 = nn.Linear(64,32)
#         self.fc7 = nn.Linear(32,12)
        
#         self.fc_bn4 = nn.BatchNorm1d(8)
#         self.fc_bn5 = nn.BatchNorm1d(64)
#         self.fc_bn6 = nn.BatchNorm1d(32)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn4(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn5(x)
#         x = F.relu(self.fc6(x))
#         x = self.fc_bn6(x)
#         x = torch.sigmoid(self.fc7(x))

#         return x


############################### 12,32,8 no batch normalizetion #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,8)
#         self.fc3 = nn.Linear(8, latent_size)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,32)
#         self.fc6 = nn.Linear(32,12)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = F.relu(self.fc5(x))
#         x = F.relu(self.fc6(x))

#         return x
    

############################### 12,32,8 with batch normalizetion #########
# class Encoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()


#         self.fc1 = nn.Linear(12,32)
#         self.fc2 = nn.Linear(32,8)
#         self.fc3 = nn.Linear(8, latent_size)
        
#         self.fc_bn1 = nn.BatchNorm1d(32)
#         self.fc_bn2 = nn.BatchNorm1d(8)

#     def forward(self, x):

#         x = F.relu(self.fc1(x))
#         x = self.fc_bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc_bn2(x)
#         x = F.relu(self.fc3(x))
#         return x


# class Decoder(nn.Module):

#     def __init__(self, latent_size):

#         super().__init__()

#         self.fc4 = nn.Linear(latent_size,8)
#         self.fc5 = nn.Linear(8,32)
#         self.fc6 = nn.Linear(32,12)
        
#         self.fc_bn3 = nn.BatchNorm1d(8)
#         self.fc_bn4 = nn.BatchNorm1d(32)


#     def forward(self, x):

#         x = F.relu(self.fc4(x))
#         x = self.fc_bn3(x)
#         x = F.relu(self.fc5(x))
#         x = self.fc_bn4(x)
#         x = F.relu(self.fc6(x))

#         return x
