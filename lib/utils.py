import numpy as np
import torch, pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from tqdm.auto import trange
from pyro.infer import Predictive

dtype = np.float32
def load_data(data_dir):
    data = {        
        'obs': np.load(f'{data_dir}/eval_lengths.npy').reshape(-1).astype(dtype),
        'AS': np.load(f'{data_dir}/sample_lengths.npy').astype(dtype),
        'BS': np.load(f'{data_dir}/beam_lengths.npy').reshape(-1).astype(dtype),
        'training': np.load(f'{data_dir}/train_lengths.npy').reshape(-1).astype(dtype),
    }
    return data

class CustomSVI:
    
    def __init__(self, data, device, dtype=torch.float):
        self.dtype = dtype
        self.device = device

        self.num_groups = 3

        if data is not None:
            yT = data['training'][:100_000]
            yGS = data['obs']
            yAS = data['AS']
            yBS = data['BS']

            self.n_train = yT.shape[0]
            # [n_t]
            self.y_T = torch.tensor(yT, dtype=dtype, device=device)
            
            self.n_val = yGS.shape[0]
            
            # [n_g]
            self.y_GS = torch.tensor(yGS, dtype=dtype, device=device)
            self.y_AS = torch.tensor(yAS, dtype=dtype, device=device)
            self.y_BS = torch.tensor(yBS, dtype=dtype, device=device)        
        else:
            self.n_train = 1
            self.n_val = 1
            self.y_T = torch.zeros([1], device=device, dtype=dtype)
            self.y_GS = torch.zeros([1], device=device, dtype=dtype)
            self.y_AS = torch.zeros([1], device=device, dtype=dtype)
            self.y_BS = torch.zeros([1], device=device, dtype=dtype)
        
        self.ready_to_train = False
        self.eps = 1e-6
        
    def p_param(self, name, value=None, constraint=None, trainable=True):
        """Create (if value is specified) or return a generative parameter"""
        if value is None:
             return pyro.param(f"theta_{name}")
        else:
            param = pyro.param(
                f"theta_{name}",
                torch.tensor(value, dtype=self.dtype, device=self.device), 
                constraint=constraint)
            return param if trainable else param.detach()
        
    def q_param(self, name, value=None, constraint=None, trainable=True):
        """Create (if value is specified) or return a variational parameter"""
        if value is None:
             return pyro.param(f"phi_{name}")
        else:
            param = pyro.param(
                f"phi_{name}",
                torch.tensor(value, dtype=self.dtype, device=self.device), 
                constraint=constraint)
            return param if trainable else param.detach()
            
    def p_dist(self, rv, num_latent_samples=None):
        raise ValueError
    
    def q_dist(self, rv, num_latent_samples=None):
        raise ValueError
        
    def obs(self):
        """Return a dictionary of observations (this is used to provide observations to svi.step)"""
        return dict()
    
    def model(self, obs=None):
        """Construct a model"""
        raise ValueError
            
    def guide(self, obs=None):
        """Construct a variational approximation"""
        raise ValueError
    
    def reset_optimizer(self, lr=0.001):
        pyro.clear_param_store()        
        self.optimizer = Adam(dict(lr=lr, betas=(0.90, 0.999)))
        self.elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=self.elbo)
        self.ll_t = []
        self.ready_to_train = True
        
    def optimize(self, num_steps, progressbar=True):
        """
        Optimise variational parameters and possibly some model parameters:
            * self.obs() is used to provide observations to model in svi.step
        """
        if not self.ready_to_train:
            self.reset_optimizer()
            
        tbar = trange(num_steps) if progressbar else range(num_steps)
        for step in tbar:
            self.ll_t.append(- self.svi.step(obs=self.obs()))
            if progressbar:
                tbar.set_postfix(dict(ELBO="{:.2f}k".format(self.ll_t[-1] / 1000)))
            elif step % 100 == 0:
                print('.', end='')
        return np.array(self.ll_t)    
    
    def predictive_samples(self, num_samples, return_sites: tuple):
        """Return a number of posterior predictive samples for specified sites"""
        predictive = Predictive(
            self.model, 
            guide=self.guide, 
            num_samples=num_samples,
            return_sites=return_sites
        )
        return predictive()
