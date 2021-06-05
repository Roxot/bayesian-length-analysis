import argparse
import torch
import numpy as np
import pyro
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

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

    # Make sure validation is disabled.
    assert pyro.__version__.startswith('1.3.0')
    pyro.enable_validation(False)

    # Clear the param store.
    pyro.clear_param_store()

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

    # Optimize the ELBO.
    elbos = hier.optimize(args.svi_steps)

    # Store parameters.
    print(f"Saving results to {args.output_folder}...")
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    pyro.get_param_store().save(output_folder / 'param-store')
    np.save(output_folder / 'elbos.npy', elbos)

    # Save ELBO plot.
    def moving_average(data_set, periods=3):
        weights = np.ones(periods) / periods
        return np.convolve(data_set, weights, mode='valid')
    plt.plot(moving_average(np.array(elbos), 30))
    plt.ylabel('ELBO')
    plt.xlabel('Steps')
    plt.title('All groups')
    plt.savefig(output_folder / 'elbos.pdf')

    # Sample lambdas.
    samples = {'lambda_T': [], 'lambda_GS': [], 'lambda_AS': [], 'lambda_BS': []}

    num_samples_100 = 50 # 5,000 samples
    
    for _ in range(num_samples_100):
        samples_i = hier.predictive_samples(
                                            100, 
                                            return_sites=(
                                            'lambda_T',
                                            'lambda_GS',
                                            'lambda_AS',
                                            'lambda_BS',)
                                            )
        for group in ['T', 'GS', 'AS', 'BS']:
            samples[f'lambda_{group}'].append(samples_i[f'lambda_{group}'].cpu().detach().numpy())
        del samples_i

    # Save posterior plots.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    hatch = {'GS': '*', 'AS': '-', 'BS': 'o', 'T': '/'}
    legend = {'GS': 'references', "AS": 'samples', "BS": 'beam search', 'T': "training"}
    for group in ['T', 'GS', 'AS', 'BS']:
        samples[f'lambda_{group}'] = np.concatenate(samples[f'lambda_{group}'])
        
    for group in ['GS', 'AS', 'BS']:#, 'T']:
        sns.distplot(
            samples[f'lambda_{group}'].mean(axis=1), 
            hist=True, kde=True, rug=False,
            hist_kws=dict(hatch=hatch[group]), kde_kws=dict(linestyle='-', shade=False), label=legend[group], ax=ax)
    ax.set_title('posterior lambda')
    ax.set_xlabel('')
    ax.set_ylabel('density')
    plt.legend()
    fig.savefig(output_folder / 'posteriors.pdf', transparent=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('--svi_steps', type=int, default=20_000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
