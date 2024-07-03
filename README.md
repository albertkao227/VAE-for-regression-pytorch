# VAE-for-regression-pytorch

Pytorch implementation of **Variational AutoEncoder For Regression: Application to Brain Aging Analysis** by Zhao et al. 

The proposed generative process explicitly models the conditional distribution of letent representations with respect to the regression target variable. Performing a variational inference procedure on this model leads to joint regularization between the VAER and a neural network regressor. The proposed approach is more accurate than state of the art models, and the disentanglement of age in latent representatoins allows for intuitive interpretation of structural development patterns of the human brain. 

- Paper https://arxiv.org/abs/1904.05948

- Repository https://github.com/QingyuZhao/VAE-for-Regression