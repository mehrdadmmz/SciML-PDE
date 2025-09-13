import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Model': ['MPP-L', 'MPP-b', 'MPP-S', 'MPP-Ti',
              'DPOT-L', 'DPOT-M', 'DPOT-S', 'DPOT-Ti', 'Hyena'],
    'Full PDE': [0.008147, 0.013481, 0.019232, 0.020492,
                 0.0347,   0.0319,   0.0349,   0.0426,   0.05562],
    'Decomposed PDE': [0.132741, 0.135356, 0.145712, 0.143235,
                       0.2081,   0.199,    0.215,    0.2116,   0.30776]
}

df = pd.DataFrame(data)


fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df['Decomposed PDE'], df['Full PDE'], s=100, marker='x')

offsets = {
    'MPP-L':   (30, 5),
    'MPP-b':   (30, 5),
    'MPP-S':   (30, 5),
    'MPP-Ti':  (-20, 10),
    'DPOT-L':  (-30, 10),
    'DPOT-M':  (-30, 5),
    'DPOT-S':  (30, 15),
    'DPOT-Ti': (15, 15),
    'Hyena': (20, 10)
}

for _, row in df.iterrows():
    dx, dy = offsets[row['Model']]
    ax.annotate(
        row['Model'],
        (row['Decomposed PDE'], row['Full PDE']),
        textcoords="offset points",
        xytext=(dx, dy),
        ha='center',
        fontsize=12
    )


ax.margins(0.2)
ax.set_xlabel('Decomposed Convection Term of \n 2D Navier Stokes (nRMSE)', fontsize=16)
ax.set_ylabel('2D Navier Stokes (nRMSE)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()


output_pdf = './NS_motivation.pdf'
plt.savefig(output_pdf, format='pdf')

plt.show()
