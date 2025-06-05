import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    'axes.grid': True,
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 100,
})


df = pd.read_csv('ns_results.csv')
mapping = {
    'fnoall':       'Baseline',
    'tsdown':       'Baseline@Spatiotemporal',
    'lr1ar07':      'Ours'
}
df['method'] = df['param'].map(mapping)
df['ds_factor'] = df['model'].str.extract(r'ds(\d+)bs').astype(int)

group = (
    df
    .groupby(['method','ds_factor'])['normalized MSE']
    .agg(['mean','std'])
    .reset_index()
)

pivot_mean = group.pivot(index='ds_factor', columns='method', values='mean').sort_index()
pivot_std  = group.pivot(index='ds_factor', columns='method', values='std').sort_index()
sim_costs = [5550, 11100, 22200, 44400, 88800, 133200, 177600]


plt.figure(figsize=(6,4))
colors = {
    'Baseline':                    '#1f77b4',
    'Baseline@Spatiotemporal':   '#ffbb78',
    'Ours':                        '#ff7f0e'
}

for method, color in colors.items():
    plt.errorbar(
        sim_costs,
        pivot_mean[method],
        yerr=pivot_std[method],
        label=method,
        color=color,
        # marker='o',
        linestyle='--',
        capsize=2
    )

plt.xscale('log')
plt.xlabel('Simulation Costs (Seconds)')
plt.ylabel('Normalized RMSE')
plt.title('2D Incompressible Navier–Stokes \n (FNO)')
plt.legend()  
plt.tight_layout()  
plt.show()


output_pdf = './random_seed_ns_FNO.pdf'
plt.savefig(output_pdf, format='pdf')