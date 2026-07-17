import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

# ==========================================
# 1. THE MATH CHEAT (Surrogate Gradient)
# ==========================================
# This defines the smooth curve used ONLY during the backward learning pass
spike_grad = surrogate.fast_sigmoid(slope=25)

# ==========================================
# 2. BUILD THE COMPILER GRAPH (Now with learning enabled)
# ==========================================
beta = 0.9

net = nn.Sequential(
    nn.Linear(784, 128),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(128, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
)

# ==========================================
# 3. SET UP THE BRAIN'S LEARNING MECHANISM
# ==========================================
# Standard PyTorch Adam optimizer to adjust the synaptic weights
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

# Rate Coding Loss: It will penalize the network if Neuron #3 doesn't fire the most
loss_fn = SF.ce_rate_loss() 

# ==========================================
# 4. CREATE THE TRAINING DATA
# ==========================================
time_steps = 25

# Create a 25ms feed of synthetic camera data [Time, Batch, Pixels]
synthetic_camera_data = torch.rand(time_steps, 1, 784) 

# The target: We want output Neuron index #3 to fire spikes
target_label = torch.tensor([3]) 

print("⚡ Starting Surrogate Gradient Descent...\n")

# ==========================================
# 5. THE TRAINING LOOP (Backpropagation through time)
# ==========================================
epochs = 15

for epoch in range(epochs):
    utils.reset(net) # Erase hardware memory before each new attempt
    
    # --- A. FORWARD PASS (Physics Simulation) ---
    spike_recording = []
    for t in range(time_steps):
        out_spikes = net(synthetic_camera_data[t])
        spike_recording.append(out_spikes)
        
    # Stack the list into a single mathematical timeline: [Time, Batch, Outputs]
    spike_timeline = torch.stack(spike_recording) 
    
    # --- B. CALCULATE ERROR (Rate Coding) ---
    loss = loss_fn(spike_timeline, target_label)
    
    # --- C. BACKWARD PASS (The Surrogate Math) ---
    optimizer.zero_grad() # Clear old gradients
    loss.backward()       # Calculate the new synapse adjustments
    optimizer.step()      # Physically change the nn.Linear weights
    
    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f}")

print("\n✅ Network successfully rewired its synapses to target Neuron #3!")


# ==========================================
# 6. THE COMPILER BRIDGE (The Translation Map)
# ==========================================
import nir
import nirtorch
import numpy as np

print("\n⚡ Building the Universal Translation Map...")

# 1. Translate a PyTorch Linear Layer into a Silicon Synapse (Affine Node)
def translate_linear(module):
    weight = module.weight.detach().numpy()
    if module.bias is not None:
        bias = module.bias.detach().numpy()
    else:
        bias = np.zeros(weight.shape[0])
    return nir.Affine(weight=weight, bias=bias)

# 2. Translate a PyTorch snnTorch Neuron into a Biological Hardware Node (LIF Node)
def translate_leaky(module):
    tau = np.array([1 / (1 - module.beta)])
    v_threshold = np.array([1.0])
    v_leak = np.array([0.0])
    r = np.array([1.0])
    return nir.LIF(tau=tau, v_threshold=v_threshold, v_leak=v_leak, r=r)

# 3. Create the FUNCTION that evaluates and maps the nodes dynamically
def compiler_map_fn(module):
    if isinstance(module, nn.Linear):
        return translate_linear(module)
    elif isinstance(module, snn.Leaky):
        return translate_leaky(module)
    return None # Safely bypasses the outer nn.Sequential container

print("⚡ Extracting the PyTorch graph into NIR hardware format...")

# 4. Extract the graph using our custom mapping function
dummy_input = torch.rand(1, 784)
nir_graph = nirtorch.extract_nir_graph(net, compiler_map_fn, dummy_input)

# 5. Freeze the graph to the hard drive
output_filename = "unc_v1_model.nir"
nir.write(output_filename, nir_graph)

print(f"\n✅ SYSTEM OVERRIDE SUCCESSFUL: '{output_filename}' has been generated.")
print("We just compiled digital code into analog hardware geometry.")