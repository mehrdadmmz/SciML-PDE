import matplotlib.pyplot as plt

plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Data for 2D Reaction-Diffusion (FNO) rollout
rollout = [1, 2, 3, 4, 5]
baseline = [0.028906, 0.033876, 0.045756, 0.059498, 0.073865]
ours = [0.023155, 0.02904, 0.040126, 0.053151, 0.066781]

     
plt.figure(figsize=(6, 4))
plt.plot(rollout, baseline,
         marker='o', linestyle='--', color='#1f77b4', label='Baseline')
plt.plot(rollout, ours,
         marker='s', linestyle='--', color='#ff7f0e', label='Ours')
plt.grid(True)

      
plt.xlabel('Rollout', fontsize=18)
plt.ylabel('Normalized RMSE', fontsize=18)
plt.title('2D Reaction-Diffusion \n (FNO)', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

  
output_pdf = './Rollout_2D_RD_FNO.pdf'
plt.savefig(output_pdf, format='pdf')
plt.show()



# Data for 2D Diffusion-Reaction (Transformer) rollout
rollout = [1, 2, 3, 4, 5]
baseline = [0.105883, 0.109151, 0.115661, 0.12328, 0.131266]
ours = [0.0602556, 0.0709661, 0.0863324, 0.102376, 0.11813]

   
plt.figure(figsize=(6, 4))
plt.plot(rollout, baseline,
         marker='o', linestyle='--', color='#1f77b4', label='Baseline')
plt.plot(rollout, ours,
         marker='s', linestyle='--', color='#ff7f0e', label='Ours')
plt.grid(True)

    
plt.xlabel('Rollout', fontsize=18)
plt.ylabel('Normalized RMSE', fontsize=18)
plt.title('2D Diffusion-Reaction \n (Transformer)', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

  
output_pdf = './Rollout_2D_RD_Transformer.pdf'
plt.savefig(output_pdf, format='pdf')
plt.show()





# Data for 2D NS (FNO) rollout
rollout = [1, 2, 3, 4, 5]
baseline = [0.048733, 0.050056, 0.067323, 0.087734, 0.10882]
ours = [0.017452, 0.025317, 0.042931, 0.060069, 0.075963]

   
plt.figure(figsize=(6, 4))
plt.plot(rollout, baseline,
         marker='o', linestyle='--', color='#1f77b4', label='Baseline')
plt.plot(rollout, ours,
         marker='s', linestyle='--', color='#ff7f0e', label='Ours')
plt.grid(True)

plt.xlabel('Rollout', fontsize=18)
plt.ylabel('Normalized RMSE', fontsize=18)
plt.title('2D Incompressible Navier–Stokes \n (FNO)', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

  
output_pdf = './Rollout_2D_NS_FNO.pdf'
plt.savefig(output_pdf, format='pdf')
plt.show()



# Data for 2D NS (Transformer) rollout
rollout = [1, 2, 3, 4, 5]
baseline = [0.047947858, 0.06525512, 0.0901043, 0.11828722, 0.14963889]
ours = [0.026561534, 0.046707958, 0.07475659, 0.106752895, 0.142262]

   
plt.figure(figsize=(6, 4))
plt.plot(rollout, baseline,
         marker='o', linestyle='--', color='#1f77b4', label='Baseline')
plt.plot(rollout, ours,
         marker='s', linestyle='--', color='#ff7f0e', label='Ours')
plt.grid(True)

plt.xlabel('Rollout', fontsize=18)
plt.ylabel('Normalized RMSE', fontsize=18)
plt.title('2D Incompressible Navier–Stokes \n (Transformer)', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

  
output_pdf = './Rollout_2D_NS_Transformer.pdf'
plt.savefig(output_pdf, format='pdf')
plt.show()


rollout = [1, 2, 3, 4, 5]
baseline = [0.067505, 0.109714, 0.150054, 0.185311, 0.218163]
ours     = [0.048125, 0.086153, 0.120555, 0.149356, 0.174979]

     
plt.figure(figsize=(6, 4))
plt.plot(rollout, baseline,
         marker='o', linestyle='--', color='#1f77b4', label='Baseline')
plt.plot(rollout, ours,
         marker='s', linestyle='--', color='#ff7f0e', label='Ours')
plt.grid(True)
plt.xlabel('Rollout', fontsize=18)
plt.ylabel('Normalized RMSE', fontsize=18)
plt.title('3D Incompressible Navier–Stokes \n (FNO)', fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

  
output_pdf = './Rollout_3D_NS_FNO.pdf'
plt.savefig(output_pdf, format='pdf')
plt.show()
