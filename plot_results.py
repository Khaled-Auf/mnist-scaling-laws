import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- CONFIGURATION ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

# Load Data
df_a = pd.read_csv("results_exp_a_data.csv")
df_b = pd.read_csv("results_exp_b_params.csv")

# Calculate Error Rates
df_a['Test_Error'] = 100 - df_a['Test_Accuracy']
df_b['Test_Error'] = 100 - df_b['Test_Accuracy']

# --- HELPER FOR PRETTY LABELS ---
def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'B'][magnitude])

# For the parameter plot, we might need decimal places (e.g., 5.8M)
def human_format_decimal(num, pos):
    if num > 1000000:
        return '%.1fM' % (num/1000000.0)
    elif num > 1000:
        return '%.0fK' % (num/1000.0)
    else:
        return str(int(num))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- PLOT 1: DATA SCALING ---
ax1.plot(df_a['Data_Size'], df_a['Test_Error'], marker='o', linestyle='-', linewidth=3, markersize=10, color='#1f77b4')
ax1.set_xscale('log')
ax1.set_yscale('log')

# Force ticks to be exactly at your data points
ax1.set_xticks(df_a['Data_Size'])
ax1.get_xaxis().set_major_formatter(FuncFormatter(human_format)) # Apply K/M formatting
ax1.minorticks_off() # Turn off the confusing small log ticks between standard values

# Labels and Titles
ax1.set_xlabel('Training Set Size (Images)', fontweight='bold')
ax1.set_ylabel('Test Error (%)', fontweight='bold')
ax1.set_title('Impact of Data Quantity', fontweight='bold')
ax1.grid(True, which="both", ls="--", alpha=0.5) # Softer grid

for x, y in zip(df_a['Data_Size'], df_a['Test_Error']):
    ax1.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0,12), ha='center', fontsize=11, fontweight='bold')

# --- PLOT 2: PARAMETER SCALING ---
ax2.plot(df_b['Num_Parameters'], df_b['Test_Error'], marker='s', linestyle='-', linewidth=3, markersize=10, color='#d62728')
ax2.set_xscale('log')
ax2.set_yscale('log')

# Force ticks to be exactly at your data points
ax2.set_xticks(df_b['Num_Parameters'])
ax2.get_xaxis().set_major_formatter(FuncFormatter(human_format_decimal))
ax2.minorticks_off()

# Labels and Titles
ax2.set_xlabel('Model Size (Parameters)', fontweight='bold')
ax2.set_ylabel('Test Error (%)', fontweight='bold')
ax2.set_title('Impact of Model Size', fontweight='bold')
ax2.grid(True, which="both", ls="--", alpha=0.5)

for x, y in zip(df_b['Num_Parameters'], df_b['Test_Error']):
    ax2.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0,12), ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Scaling Laws in Neural Networks (MNIST Dataset)', fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('scaling_laws_poster_final.png', dpi=300, bbox_inches='tight')
print("Final poster plots saved to 'scaling_laws_poster_final.png'")
plt.show()