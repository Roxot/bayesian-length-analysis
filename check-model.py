import argparse
import torch
import numpy as np
import pyro

from lib import poisson_predictive_checks
from lib import load_data, HierarchicalGammaPoisson

def main(args):
    device = torch.device(args.device)

    print("Loading data...")
    data = load_data(args.data_folder)
    yT = data['training']
    yGS = data['obs']
    yAS = data['AS']
    yBS = data['BS']
    print(f"T shape {yT.shape} mean {np.mean(yT)} std {np.std(yT)}") 
    print(f"GS shape {yGS.shape} mean {np.mean(yGS)} std {np.std(yGS)}")
    print(f"AS shape {yAS.shape} mean {np.mean(yAS)} std {np.std(yAS)}")
    print(f"BS shape {yBS.shape} mean {np.mean(yBS)} std {np.std(yBS)}")
    all_data = {
        "T": yT,
        "GS": yGS,
        "AS": yAS,
        "BS": yBS 
    }

    print("Loading model parameters...")
    hier = HierarchicalGammaPoisson(
        data,
        model_val=True, 
        use_mu=True,
        empirical_bayes=False,
        prior_params=dict(
            alpha_T_r=1.0,
            beta_T_r=10.0,
        ),
        device=device
    )
    pyro.clear_param_store()
    pyro.get_param_store().load(args.param_store)

    print("Sampling from the predictive posterior...")
    samples = hier.predictive_samples(
        1000, 
        return_sites=(
            'y_T', 'y_AS', 'y_BS', 'y_GS',
        )
    )

    print("Running predictive checks...")
    for group in ["T", "GS", "AS", "BS"]:
        print(f'# {group}')
        print(poisson_predictive_checks(samples[f'y_{group}'].detach().cpu().numpy(), 
                                        all_data[group]))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('param_store', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
