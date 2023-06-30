import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from torchvision.models import resnet18, resnet50, resnet152
import numpy as np
import os
import subprocess
import sys

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument("-v", "--model_version", type=int, default=18, choices=[18, 50, 152], help="Model version: 18, 50 or 152")
parser.add_argument("-b", "--batch_size", type=int, default=128, choices=[32, 64, 128, 256, 512], help="Batch size: 32, 64, 128, 256 or 512")
parser.add_argument("-n", "--num_runs", type=int, default=5, help="Number of runs: any positive integer")

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_version = args.model_version
batch_size = args.batch_size
num_runs = args.num_runs

if 'SLURM_NODELIST' in os.environ:
    nodelist = os.environ['SLURM_NODELIST']
    # Extract the GPU node name
    gpu_node = nodelist.split()[0]

# Check if CUDA is available and print the number of GPUs available
if torch.cuda.is_available():
    print("Number of GPUs available: ", torch.cuda.device_count(), gpu_node)
else:
    print("CUDA is not available. Training will be performed on CPU.")

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations for the training and test set
transform = transforms.Compose(
    [transforms.Resize(224),  # Resizing to 224x224 for ResNet50
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the pre-downloaded CIFAR10 training set
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Define the classes in the CIFAR10 dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Load the ResNet model based on the specified version
if model_version == 18:
    net = resnet18(weights=None)
elif model_version == 50:
    net = resnet50(weights=None)
else: # model_version == 152
    net = resnet152(weights=None)

net = net.to(device)  # Move the network to GPU if available

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Initialize the lists for storing:
losses = []  		# loss values
losses_run = []  	# loss values per run
times = [] 		# all time values
times_run = []  	# mean time values per run

for run in range(num_runs):
    times_epoch = []  # times values per epoch
    losses_epoch = []  # loss values per epoch

    # Train the network
    for epoch in range(20):

        # Collect data from nvidi-smi
        lines = subprocess.check_output(['nvidia-smi', '-q']).decode(sys.stdout.encoding).split(os.linesep)

        sm = []
        temp = []
        
        for line in lines:
            if 'SM                                :' in line:
                value = line.strip().split(':')[1].strip()
                sm.append(value)
            elif 'GPU Current Temp' in line:
                temp = line.strip().split(':')[1].strip()
        
        print('Run [%d] Epoch [%d] Processor: %s' % (run, epoch + 1, sm[0]))
        print('Run [%d] Epoch [%d] Processor Max: %s' % (run, epoch + 1, sm[1]))
        print('Run [%d] Epoch [%d] Temp: %s' % (run, epoch + 1, temp))

        running_loss = 0.0
        start_time = time.time()  # Start timing
     
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        end_time = time.time()  # End timing
        avg_loss = running_loss / (i + 1)
        elapsed_time = end_time - start_time
        print('Run [%d] Epoch [%d] loss: %.3f' % (run, epoch + 1, avg_loss))
        print('Run [%d] Epoch [%d] training time: %.3f seconds' % (run, epoch + 1, elapsed_time))
        losses_epoch.append(avg_loss)  # Append the loss value to losses
        times_epoch.append(elapsed_time)  # Append the time value to times

    times.append(times_epoch)
    times_run.append(np.mean(np.array(times_epoch)))

    losses.append(losses_epoch)
    losses_run.append(np.mean(np.array(losses_epoch)))

# Convert losses and times list to numpy array for efficient computation
losses = np.array(losses)
times_epoch = np.array(times)


# Compute the average and standard deviation of losses
average_loss = np.mean(losses)
std_dev_loss = np.std(losses_run)

# Compute the average and standard deviation of times
average_time = np.mean(times)
std_dev_time = np.std(times_run)

print('Finished Training')
print('Average loss over %d runs: %.3f' % (num_runs, average_loss))
print('Standard deviation of loss over %d runs: %.3f' % (num_runs, std_dev_loss))
print('Average training time over %d runs: %.3f seconds' % (num_runs, average_time))
print('Standard deviation of training time over %d runs: %.3f seconds' % (num_runs, std_dev_time))