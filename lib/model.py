import torch
import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints

from .utils import CustomSVI

class HierarchicalGammaPoisson(CustomSVI):
    
    def __init__(self, data, model_val, use_mu, empirical_bayes, device, dtype=torch.float, prior_params=dict()):
        super().__init__(data, device, dtype)
        self.model_val = model_val
        self.use_mu = use_mu
        self.empirical_bayes = empirical_bayes
        self.prior_params = prior_params
            
    def p_dist(self, rv, num_latent_samples=None):
        if rv == 'alpha_T':
            return dist.Exponential(self.p_param('alpha_T_r'))
        if rv == 'beta_T':
            return dist.Exponential(self.p_param('beta_T_r'))
        if self.model_val and self.use_mu:
            if rv == 'eta':
                return dist.Exponential(self.p_param('eta_r'))
        raise ValueError
        
    def q_dist(self, rv, num_latent_samples=None):
        if rv == 'alpha_T':
            return dist.Gamma(self.q_param('alpha_T_a'), self.q_param('alpha_T_b'))
        if rv == 'beta_T':
            return dist.Gamma(self.q_param('beta_T_a'), self.q_param('beta_T_b'))
        if self.model_val and self.use_mu:
            if rv == 'eta':
                return dist.Gamma(self.q_param('eta_a'), self.q_param('eta_b'))
    
    def obs(self):
        if self.model_val:
            return dict(y_T=self.y_T, y_GS=self.y_GS, y_AS=self.y_AS, y_BS=self.y_BS)
        else:
            return dict(y_T=self.y_T)
    
    def model(self, obs=None):    
        alpha_T = pyro.sample(
            'alpha_T', 
            dist.Exponential(
                self.p_param('alpha_T_r', self.prior_params.get('alpha_T_r', 1.), 
                    constraints.positive, trainable=self.empirical_bayes))
        )
        beta_T = pyro.sample(
            'beta_T', 
            dist.Exponential(
                self.p_param('beta_T_r', self.prior_params.get('beta_T_r', 1.), 
                    constraints.positive, trainable=self.empirical_bayes))
        )
        
        with pyro.plate("training", self.n_train):
            lambda_T = pyro.sample(
                "lambda_T", 
                dist.Gamma(
                    alpha_T * torch.ones(self.n_train, dtype=self.dtype, device=self.device), 
                    beta_T * torch.ones(self.n_train, dtype=self.dtype, device=self.device)
                )
            )
            pyro.sample("y_T", dist.Poisson(lambda_T), obs=obs['y_T'] if obs else None)    
        
        if self.model_val:
            alphas = dict()
            betas = dict()
            
            if self.use_mu:
                mu_t = dist.Gamma(alpha_T, beta_T).mean
                eta = pyro.sample(
                        'eta', 
                        dist.Exponential(
                            self.p_param('eta_r', self.prior_params.get('eta_r', 1.), 
                                constraints.positive, trainable=self.empirical_bayes))
                    )
            else:
                mu_t, eta = None, None
                
            for group_name in ['GS', 'AS', 'BS']:                                                            
                if self.use_mu:                    
                    alphas[group_name] = pyro.sample(
                        f'alpha_{group_name}', 
                        dist.Exponential(eta)
                    )
                    # beta_g = mu_t
                    betas[group_name] = 1. / mu_t
                else:
                    alphas[group_name] = alpha_T
                    betas[group_name] = beta_T
            
            with pyro.plate("validation", self.n_val):                
                for group_name in ['GS', 'AS', 'BS']:                                                                                
                    lambda_g = pyro.sample(
                        f"lambda_{group_name}", 
                        dist.Gamma(
                            alphas[group_name] * torch.ones(self.n_val, dtype=self.dtype, device=self.device), 
                            betas[group_name] * torch.ones(self.n_val, dtype=self.dtype, device=self.device)
                        )
                    )
                    pyro.sample(
                        f"y_{group_name}", 
                        dist.Poisson(lambda_g), 
                        obs=obs[f"y_{group_name}"] if obs else None
                    )    
            
    def guide(self, obs=None):
        # q(alpha_T)
        alpha_T = pyro.sample(
            'alpha_T',
            dist.Gamma(
                self.q_param('alpha_T_a', 1.0, constraints.positive),
                self.q_param('alpha_T_b', 1.0, constraints.positive)
            )            
        )
        # q(beta_T)
        beta_T = pyro.sample(
            'beta_T',
            dist.Gamma(
                self.q_param('beta_T_a', 1.0, constraints.positive),
                self.q_param('beta_T_b', 1.0, constraints.positive)
            )            
        )
        with pyro.plate("training", self.n_train):
            alpha_T_prime = alpha_T + self.y_T
            beta_T_prime = beta_T + torch.ones(self.n_train, dtype=self.dtype, device=self.device)
            lambda_T = pyro.sample("lambda_T", dist.Gamma(alpha_T_prime, beta_T_prime))
        
        if self.model_val:
            if self.use_mu:
                eta = pyro.sample(
                    'eta',
                    dist.Gamma(
                        self.q_param('eta_a', 1.0, constraints.positive),
                        self.q_param('eta_b', 1.0, constraints.positive)
                    )            
                )
                alphas = dict()
                for group_name in ['GS', 'AS', 'BS']:
                    if self.use_mu:                
                        alphas[group_name] = pyro.sample(
                            f'alpha_{group_name}',
                            dist.Gamma(
                                self.q_param(f'alpha_{group_name}_a', 1.0, constraints.positive),
                                self.q_param(f'alpha_{group_name}_b', 1.0, constraints.positive)
                            )            
                        ) 
                
            with pyro.plate("validation", self.n_val):
                for group_name, y_group in [('GS', self.y_GS), ('AS', self.y_AS), ('BS', self.y_BS)]:
                    if self.use_mu:
                        alpha_g = alphas[group_name] + y_group
                        beta_g = beta_T / alpha_T + torch.ones(self.n_val, dtype=self.dtype, device=self.device)
                    else:
                        alpha_g = alpha_T + y_group
                        beta_g = beta_T + torch.ones(self.n_val, dtype=self.dtype, device=self.device)
                    lambda_g = pyro.sample(f"lambda_{group_name}", dist.Gamma(alpha_g, beta_g))
