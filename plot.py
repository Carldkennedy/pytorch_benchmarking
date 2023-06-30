import pandas as pd
import matplotlib.pyplot as plt

data_file = "stats.out"  
y_axis = "shared"         # options: shared; zero; none
one_figure = "no"         # If "yes" all plots on same figure. Don't combine with shared y-axis
# Read the data from the file
data = []
with open(data_file, "r") as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split()
            gpu_and_cuda = parts[0].split("_")
            gpu_cuda = "_".join(gpu_and_cuda[0:2])
            batch_size = gpu_and_cuda[3]
            model = gpu_and_cuda[2]
            avg_time = float(parts[1])
            std_dev = float(parts[2])
            data.append([gpu_cuda, model, batch_size, avg_time, std_dev])

# Create a DataFrame
columns = ["GPU_Cuda", "ResNetModel", "BatchSize", "AverageTime", "StdDev"]
df = pd.DataFrame(data, columns=columns)

# Convert 'BatchSize' column to integer
df['BatchSize'] = df['BatchSize'].astype(int)
# Set shared y-axis 
y_min_shared = df['AverageTime'].min() * 0.95
y_max_shared = df['AverageTime'].max() * 1.05
        
# Define marker styles and colors for ResNet models and GPUs
markers = {'11.8': 'o', '12.1': 'v'}
colors = {'A': 'red', 'H': 'black'}

# Plotting the data for each ResNet model
XLABEL = 'Batch Size'
YLABEL = 'Mean Training Time (s)'
TITLE = 'PyTorch Benchmarking with ResNet'

ANCHOR = [0.5,1]

resnet_models = df['ResNetModel'].unique()

x_min = df['BatchSize'].min() - df['BatchSize'].min()
x_max = df['BatchSize'].max() + df['BatchSize'].min()
if one_figure == "yes":
    fig, ax = plt.subplots()

for model in resnet_models:
    model_data = df[df['ResNetModel'] == model]
    unique_gpu_count = model_data['GPU_Cuda'].nunique()
    print(unique_gpu_count)
    if one_figure != "yes":
        fig, ax = plt.subplots()
    
    for gpu_cuda in model_data['GPU_Cuda'].unique():
        gpu_data = model_data[model_data['GPU_Cuda'] == gpu_cuda]
        batch_sizes = gpu_data['BatchSize']
        times = gpu_data['AverageTime']
        std_devs = gpu_data['StdDev']
        color = colors[gpu_cuda[0]] # GPU identifier
        marker = markers[gpu_cuda.split("_")[1]] # cuda version
        label = f'{gpu_cuda.split("_")[0]} - Cuda {gpu_cuda.split("_")[1]}'
        ax.errorbar(batch_sizes, times, yerr=std_devs, fmt=marker, color=color, label=label, capsize=6, capthick=1, elinewidth=1, zorder=2)
        
   
    # Set x-y-axis limits

    if y_axis == "zero":
        y_min = 0
        ax.set_ylim(y_min)   

    if y_axis == "shared":
        y_min = y_min_shared
        y_max = y_max_shared
        ax.set_ylim(y_min, y_max)    

    ax.set_xlim(x_min, x_max)

    if model == "152" and one_figure != "yes":
        # Define the coordinates for the text
        x_pos = 512  # Adjust the x-coordinate as needed
        y_pos = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 #y_min  # Adjust the y-coordinate as needed
        # Add the "Out of memory" text vertically
        ax.text(x_pos, y_pos, "Out of memory", rotation="vertical", va="center", ha="center", color="red")

    # Set x-axis markers
    x_markers = [32, 64, 128, 256, 512]
    ax.set_xticks(x_markers)            

    # Axis settings
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_title(f'ResNet{model} trained on CIFAR10')
    
    if one_figure != "yes":
        ax.legend()
        # Adjust the legend position
        legend = ax.legend(loc='upper left', bbox_to_anchor=(ANCHOR[0],ANCHOR[1]))
        # Save the plot to a file (e.g., PNG format)
        fig.savefig(f'ResNet{model}_{y_axis}.png', dpi=300)

if one_figure == "yes":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:unique_gpu_count], labels[:unique_gpu_count], loc='upper left', bbox_to_anchor=(ANCHOR[0], ANCHOR[1]))
        ax.set_title(f'ResNet models trained on CIFAR10')
        # Save the plot to a file (e.g., PNG format)
        fig.savefig(f'ResNet_all_{y_axis}.png', dpi=300)