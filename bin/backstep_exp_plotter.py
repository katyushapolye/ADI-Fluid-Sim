import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Re = 1000
U0 = 180


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12/2, 18/2))  # Two plots side-by-side
fig.suptitle("U Velocity Profiles At Re = {}".format(Re), fontsize=12, y=0.99,x=0.53)

axes[0].set_xlim(-20, U0)
axes[1].set_xlim(-20, U0)

axes[0].set_ylim(0, 10)
axes[1].set_ylim(0, 10)

xS = 4.00

# Common experimental data
exp_data = pd.read_csv('Graphs/Backwards_Step_re{}/expData_re={}_{}.csv'.format(Re,Re,xS))
exp_data.columns = exp_data.columns.str.strip()
upar = exp_data['x'].to_list()
yu_exp = exp_data['y'].to_list()
axes[0].scatter(upar, yu_exp, facecolor='none', edgecolor='black', label='Armaly et al.', marker='o', s=100)

# Plot 1: Using Numerical Data (N=16)
data = pd.read_csv('Graphs/Backwards_Step_re{}/64/UField_Stable_Backstep_re={}.csv'.format(Re,Re))
yu = data.iloc[:, 0].to_list()
jIndex_16 = int((((xS*0.5) + 0.5) /(yu[1] - yu[0]))) 
xu = data.iloc[:, jIndex_16].to_list()
axes[0].plot(np.array(xu) * U0, np.array(yu) * 10, label='ADI - N = 64')


data = pd.read_csv('Graphs/Backwards_Step_re{}/32/UField_Stable_Backstep_re={}.csv'.format(Re,Re))
yu = data.iloc[:, 0].to_list()
jIndex_16 = int((((xS*0.5) + 0.5) /(yu[1] - yu[0]))) 
xu = data.iloc[:, jIndex_16].to_list()
axes[0].plot(np.array(xu) * U0, np.array(yu) * 10, label='ADI - N = 32')


axes[0].set_title("x/S = {}".format(xS))
axes[0].set_xlabel("u")
axes[0].set_ylabel("y")
axes[0].legend()


##plot 2
#
xS = 14.0


# Common experimental data
exp_data = pd.read_csv('Graphs/Backwards_Step_re{}/expData_re={}_{}.csv'.format(Re,Re,xS))
exp_data.columns = exp_data.columns.str.strip()
upar = exp_data['x'].to_list()
yu_exp = exp_data['y'].to_list()
axes[1].scatter(upar, yu_exp, facecolor='none', edgecolor='black', label='Armaly et al.', marker='o', s=100)

# Plot 1: Using Numerical Data (N=16)
data = pd.read_csv('Graphs/Backwards_Step_re{}/64/UField_Stable_Backstep_re={}.csv'.format(Re,Re))
yu = data.iloc[:, 0].to_list()
jIndex_16 = int((((xS*0.5) + 0.5) /(yu[1] - yu[0]))) 
xu = data.iloc[:, jIndex_16].to_list()
axes[1].plot(np.array(xu) * U0, np.array(yu) * 10, label='ADI - N = 64')


data = pd.read_csv('Graphs/Backwards_Step_re{}/32/UField_Stable_Backstep_re={}.csv'.format(Re,Re))
yu = data.iloc[:, 0].to_list()
jIndex_16 = int((((xS*0.5) + 0.5) /(yu[1] - yu[0]))) 
xu = data.iloc[:, jIndex_16].to_list()
axes[1].plot(np.array(xu) * U0, np.array(yu) * 10, label='ADI - N = 32')


axes[1].set_title("x/S = {}".format(xS))
axes[1].set_xlabel("u")
axes[1].set_ylabel("y")
axes[1].legend()







# Adjust layout and show
plt.tight_layout()
plt.show()
