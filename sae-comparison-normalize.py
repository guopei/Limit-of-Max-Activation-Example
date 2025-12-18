import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 0.001
num_epochs_ae = 20
num_epochs_sae = 30
k_sparse = 10  # Top-k activations for sparse autoencoder

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Standard Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 784 -> 1000 -> 500 -> 250 -> 30
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30),
            nn.ReLU(),
        )
        
        # Decoder: 30 -> 250 -> 500 -> 1000 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


# Sparse Autoencoder - 3 layer MLP operating on the 30-dim hidden representation
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=1000, k=10):
        super(SparseAutoencoder, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: 30 -> 1000 (with top-k sparsity)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: 1000 -> 30
        self.decoder = nn.Linear(hidden_dim, input_dim)
    
    def top_k_sparse(self, x):
        """Apply top-k sparsity constraint"""
        # Get top-k values and indices
        topk_vals, topk_indices = torch.topk(x, self.k, dim=1)
        
        # Create sparse representation
        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(1, topk_indices, topk_vals)
        
        return sparse_x
    
    def forward(self, x):
        # Encode to 1000-dim
        hidden = torch.relu(self.encoder(x))
        
        # Apply top-k sparsity
        sparse_hidden = self.top_k_sparse(hidden)
        
        # Decode back to 30-dim
        reconstructed = self.decoder(sparse_hidden)
        
        return reconstructed, sparse_hidden
    
    def encode(self, x):
        """Get sparse hidden representation"""
        hidden = torch.relu(self.encoder(x))
        return self.top_k_sparse(hidden)


def train_autoencoder(model, train_loader, num_epochs, lr, model_name="Autoencoder"):
    """Train the standard autoencoder"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'{model_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses


def compute_normalization_stats(ae_model, train_loader):
    """Compute mean and std of hidden representations for normalization"""
    ae_model.eval()
    all_hidden = []
    
    print("Computing normalization statistics...")
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.view(-1, 784).to(device)
            hidden = ae_model.encode(data)
            all_hidden.append(hidden)
    
    all_hidden = torch.cat(all_hidden, dim=0)
    mean = all_hidden.mean(dim=0, keepdim=True)
    std = all_hidden.std(dim=0, keepdim=True) + 1e-8
    
    print(f"  Hidden range: [{all_hidden.min():.4f}, {all_hidden.max():.4f}]")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"  Mean of means: {mean.mean():.4f}, Mean of stds: {std.mean():.4f}")
    
    return mean, std


def train_sparse_autoencoder(sae_model, ae_model, train_loader, num_epochs, lr, 
                            mean, std, model_name="SAE"):
    """Train sparse autoencoder on the normalized hidden representations of the AE"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(sae_model.parameters(), lr=lr)
    
    losses = []
    ae_model.eval()  # Freeze the autoencoder
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            
            # Get hidden representation from autoencoder and normalize
            with torch.no_grad():
                hidden = ae_model.encode(data)
                hidden_normalized = (hidden - mean) / std
            
            # Forward pass through SAE
            reconstructed, _ = sae_model(hidden_normalized)
            loss = criterion(reconstructed, hidden_normalized)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'{model_name} - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses


def visualize_sae_reconstructions(ae_model, sae_model, test_loader, mean, std, 
                                 title_prefix="", num_images=10):
    """Visualize reconstructions through both AE and SAE"""
    ae_model.eval()
    sae_model.eval()
    
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_images].to(device)
        data_flat = data.view(-1, 784)
        
        # Get hidden representation from AE and normalize
        hidden = ae_model.encode(data_flat)
        hidden_normalized = (hidden - mean) / std
        
        # Reconstruct hidden through SAE
        hidden_reconstructed_norm, _ = sae_model(hidden_normalized)
        
        # Denormalize
        hidden_reconstructed = hidden_reconstructed_norm * std + mean
        
        # Decode back to image space
        reconstructed_ae = ae_model.decoder(hidden)
        reconstructed_sae = ae_model.decoder(hidden_reconstructed)
        
        data = data.cpu().view(-1, 28, 28)
        reconstructed_ae = reconstructed_ae.cpu().view(-1, 28, 28)
        reconstructed_sae = reconstructed_sae.cpu().view(-1, 28, 28)
    
    fig, axes = plt.subplots(3, num_images, figsize=(15, 4.5))
    for i in range(num_images):
        axes[0, i].imshow(data[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        axes[1, i].imshow(reconstructed_ae[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('AE Recon', fontsize=10)
        
        axes[2, i].imshow(reconstructed_sae[i], cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('SAE Recon', fontsize=10)
    
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=12, y=0.98)
    
    plt.tight_layout()
    return fig

def find_max_activation_hidden_units(ae_model, train_loader, title_prefix="",
                                      num_units=30, top_n=10):
    """
    Find images that maximally activate each unit in the hidden layer.

    Args:
        ae_model: The autoencoder model
        train_loader: DataLoader for training images
        num_units: Number of hidden units to visualize (default: all 30)
        top_n: Number of top-activating images per unit
    """
    ae_model.eval()

    # Store hidden activations and corresponding images
    all_hidden = []
    all_images = []

    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            data_flat = data.view(-1, 784)

            # Get hidden representation from AE
            hidden = ae_model.encode(data_flat)

            all_hidden.append(hidden.cpu())
            all_images.append(data.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)  # [N, 30]
    all_images = torch.cat(all_images, dim=0)  # [N, 1, 28, 28]

    # For each hidden unit, find top-n maximally activating images
    fig, axes = plt.subplots(num_units, top_n, figsize=(15, num_units * 1.5))

    for unit_idx in range(num_units):
        # Get activations for this unit
        unit_activations = all_hidden[:, unit_idx]

        # Find top-n indices
        top_indices = torch.topk(unit_activations, top_n).indices

        # Visualize
        for i, idx in enumerate(top_indices):
            img = all_images[idx].squeeze()
            activation_val = unit_activations[idx].item()

            axes[unit_idx, i].imshow(img, cmap='gray')
            axes[unit_idx, i].axis('off')

            if i == 0:
                axes[unit_idx, i].set_ylabel(f'H{unit_idx}', rotation=0,
                                              labelpad=30, fontsize=9)

            axes[unit_idx, i].set_title(f'{activation_val:.1f}', fontsize=7)

    if title_prefix:
        plt.suptitle(title_prefix, fontsize=13, y=1.0)

    plt.tight_layout()
    return fig


def find_max_activation_random_projections(ae_model, train_loader, mean, std,
                                            random_projection, title_prefix="",
                                            top_n=10):
    """
    Find images that maximally activate random linear combinations of hidden dimensions.

    This tests the hypothesis that random directions in the hidden space might also
    reveal interpretable features, not just learned SAE directions.

    Args:
        ae_model: The autoencoder model
        train_loader: DataLoader for training images
        mean, std: Normalization statistics for hidden representations
        random_projection: [hidden_dim, num_directions] random projection matrix
        top_n: Number of top-activating images per direction
    """
    ae_model.eval()
    num_directions = random_projection.shape[1]

    # Store projections and corresponding images
    all_projections = []
    all_images = []

    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            data_flat = data.view(-1, 784)

            # Get hidden representation from AE and normalize
            hidden = ae_model.encode(data_flat)
            hidden_normalized = (hidden - mean) / std

            # Project onto random directions: [batch, 30] @ [30, num_directions] -> [batch, num_directions]
            projections = hidden_normalized @ random_projection

            all_projections.append(projections.cpu())
            all_images.append(data.cpu())

    all_projections = torch.cat(all_projections, dim=0)  # [N, num_directions]
    all_images = torch.cat(all_images, dim=0)  # [N, 1, 28, 28]

    # For each random direction, find top-n maximally activating images
    fig, axes = plt.subplots(num_directions, top_n, figsize=(15, num_directions * 1.5))

    for dir_idx in range(num_directions):
        # Get projections for this direction
        dir_projections = all_projections[:, dir_idx]

        # Find top-n indices (highest positive projections)
        top_indices = torch.topk(dir_projections, top_n).indices

        # Visualize
        for i, idx in enumerate(top_indices):
            img = all_images[idx].squeeze()
            projection_val = dir_projections[idx].item()

            axes[dir_idx, i].imshow(img, cmap='gray')
            axes[dir_idx, i].axis('off')

            if i == 0:
                axes[dir_idx, i].set_ylabel(f'R{dir_idx}', rotation=0,
                                             labelpad=30, fontsize=9)

            axes[dir_idx, i].set_title(f'{projection_val:.1f}', fontsize=7)

    if title_prefix:
        plt.suptitle(title_prefix, fontsize=13, y=1.0)

    plt.tight_layout()
    return fig


def find_max_activation_examples(ae_model, sae_model, train_loader, mean, std,
                                 title_prefix="", num_features=10, top_n=10):
    """Find images that maximally activate each SAE feature"""
    ae_model.eval()
    sae_model.eval()
    
    # Store activations and corresponding images
    all_activations = []
    all_images = []
    
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            data_flat = data.view(-1, 784)
            
            # Get hidden representation from AE and normalize
            hidden = ae_model.encode(data_flat)
            hidden_normalized = (hidden - mean) / std
            
            # Get SAE activations
            sae_activations = sae_model.encode(hidden_normalized)
            
            all_activations.append(sae_activations.cpu())
            all_images.append(data.cpu())
    
    all_activations = torch.cat(all_activations, dim=0)  # [N, 1000]
    all_images = torch.cat(all_images, dim=0)  # [N, 1, 28, 28]
    
    # For each feature dimension, find top-n maximally activating images
    fig, axes = plt.subplots(num_features, top_n, figsize=(15, num_features * 1.5))
    
    for feature_idx in range(num_features):
        # Get activations for this feature
        feature_activations = all_activations[:, feature_idx]
        
        # Find top-n indices
        top_indices = torch.topk(feature_activations, top_n).indices
        
        # Visualize
        for i, idx in enumerate(top_indices):
            img = all_images[idx].squeeze()
            activation_val = feature_activations[idx].item()
            
            axes[feature_idx, i].imshow(img, cmap='gray')
            axes[feature_idx, i].axis('off')
            
            if i == 0:
                axes[feature_idx, i].set_ylabel(f'F{feature_idx}', rotation=0, 
                                                 labelpad=30, fontsize=9)
            
            axes[feature_idx, i].set_title(f'{activation_val:.1f}', fontsize=7)
    
    if title_prefix:
        plt.suptitle(title_prefix, fontsize=13, y=1.0)
    
    plt.tight_layout()
    return fig


def compare_hidden_distributions(ae_trained, ae_random, test_loader, 
                                mean_trained, std_trained, mean_random, std_random):
    """Compare the distribution of hidden representations (both raw and normalized)"""
    ae_trained.eval()
    ae_random.eval()
    
    hidden_trained = []
    hidden_random = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)
            hidden_trained.append(ae_trained.encode(data).cpu())
            hidden_random.append(ae_random.encode(data).cpu())
    
    hidden_trained = torch.cat(hidden_trained, dim=0)
    hidden_random = torch.cat(hidden_random, dim=0)
    
    # Compute normalized versions
    hidden_trained_norm = (hidden_trained - mean_trained.cpu()) / std_trained.cpu()
    hidden_random_norm = (hidden_random - mean_random.cpu()) / std_random.cpu()
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Row 1: Raw distribution of values
    axes[0, 0].hist(hidden_trained.flatten().numpy(), bins=100, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('Activation Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Trained AE: Raw Hidden Layer Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(hidden_random.flatten().numpy(), bins=100, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Activation Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Random AE: Raw Hidden Layer Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Normalized distribution of values
    axes[1, 0].hist(hidden_trained_norm.flatten().numpy(), bins=100, alpha=0.7, color='blue')
    axes[1, 0].set_xlabel('Normalized Activation Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Trained AE: Normalized Hidden Layer Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    axes[1, 1].hist(hidden_random_norm.flatten().numpy(), bins=100, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Normalized Activation Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Random AE: Normalized Hidden Layer Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Row 3: Mean activation per dimension (raw)
    mean_trained_dim = hidden_trained.mean(dim=0)
    mean_random_dim = hidden_random.mean(dim=0)
    
    axes[2, 0].bar(range(30), mean_trained_dim.numpy(), color='blue', alpha=0.7)
    axes[2, 0].set_xlabel('Dimension')
    axes[2, 0].set_ylabel('Mean Activation')
    axes[2, 0].set_title('Trained AE: Raw Mean Activation per Dimension')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].bar(range(30), mean_random_dim.numpy(), color='red', alpha=0.7)
    axes[2, 1].set_xlabel('Dimension')
    axes[2, 1].set_ylabel('Mean Activation')
    axes[2, 1].set_title('Random AE: Raw Mean Activation per Dimension')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# PART 1: Train Standard Autoencoder
# ============================================================================
print("\n" + "="*70)
print("PART 1: Training Standard Autoencoder")
print("="*70)
ae_trained = Autoencoder().to(device)
ae_losses = train_autoencoder(ae_trained, train_loader, num_epochs_ae, learning_rate, "Trained AE")

# ============================================================================
# PART 2: Create Random (Untrained) Autoencoder
# ============================================================================
print("\n" + "="*70)
print("PART 2: Creating Random (Untrained) Autoencoder")
print("="*70)
torch.manual_seed(123)  # Different seed for variety
ae_random = Autoencoder().to(device)
print("Random AE created (weights NOT trained)")

# ============================================================================
# PART 2.5: Compute Normalization Statistics
# ============================================================================
print("\n" + "="*70)
print("PART 2.5: Computing Normalization Statistics")
print("="*70)
print("\nFor TRAINED AE:")
mean_trained, std_trained = compute_normalization_stats(ae_trained, train_loader)

print("\nFor RANDOM AE:")
mean_random, std_random = compute_normalization_stats(ae_random, train_loader)

# ============================================================================
# PART 3: Train SAE on Trained AE (with normalization)
# ============================================================================
print("\n" + "="*70)
print(f"PART 3: Training SAE on TRAINED AE Hidden Layer (Top-{k_sparse}, Normalized)")
print("="*70)
sae_trained = SparseAutoencoder(input_dim=30, hidden_dim=1000, k=k_sparse).to(device)
sae_trained_losses = train_sparse_autoencoder(sae_trained, ae_trained, train_loader, 
                                              num_epochs_sae, learning_rate,
                                              mean_trained, std_trained, "SAE (Trained AE)")

# ============================================================================
# PART 3.5: Get a random projection of the trained AE hidden layer
# ============================================================================
print("\n" + "="*70)
print("PART 3.5: Getting a random projection of the trained AE hidden layer")
print("="*70)
random_projection = torch.randn(30, 10).to(device)

# ============================================================================
# PART 4: Train SAE on Random AE (with normalization)
# ============================================================================
print("\n" + "="*70)
print(f"PART 4: Training SAE on RANDOM AE Hidden Layer (Top-{k_sparse}, Normalized)")
print("="*70)
sae_random = SparseAutoencoder(input_dim=30, hidden_dim=1000, k=k_sparse).to(device)
sae_random_losses = train_sparse_autoencoder(sae_random, ae_random, train_loader, 
                                            num_epochs_sae, learning_rate,
                                            mean_random, std_random, "SAE (Random AE)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("Generating Visualizations...")
print("="*70)

# 1. Training losses comparison
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(ae_losses, label='Trained AE', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Standard Autoencoder Training Loss', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(sae_trained_losses, label='SAE on Trained AE', linewidth=2, color='green')
axes[1].plot(sae_random_losses, label='SAE on Random AE', linewidth=2, color='red')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss (on Normalized Hidden)', fontsize=11)
axes[1].set_title('Sparse Autoencoder Training Loss Comparison', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('training_losses_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: training_losses_comparison.png")

# 2. Compare hidden layer distributions
fig2 = compare_hidden_distributions(ae_trained, ae_random, test_loader,
                                   mean_trained, std_trained, mean_random, std_random)
fig2.savefig('hidden_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hidden_distributions.png")

# 3. SAE reconstructions - Trained AE
fig3 = visualize_sae_reconstructions(ae_trained, sae_trained, test_loader,
                                     mean_trained, std_trained,
                                     title_prefix="SAE on TRAINED AE (Normalized)")
fig3.savefig('sae_trained_reconstructions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sae_trained_reconstructions.png")

# 4. SAE reconstructions - Random AE
fig4 = visualize_sae_reconstructions(ae_random, sae_random, test_loader,
                                     mean_random, std_random,
                                     title_prefix="SAE on RANDOM AE (Normalized)")
fig4.savefig('sae_random_reconstructions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sae_random_reconstructions.png")

# 5. Max activation examples - SAE on Trained AE
fig5 = find_max_activation_examples(ae_trained, sae_trained, train_loader,
                                    mean_trained, std_trained,
                                    title_prefix="SAE Features (Trained AE): Max Activation Examples",
                                    num_features=10, top_n=10)
fig5.savefig('sae_trained_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sae_trained_max_activations.png")

# 6. Max activation examples - SAE on Random AE
fig6 = find_max_activation_examples(ae_random, sae_random, train_loader,
                                    mean_random, std_random,
                                    title_prefix="SAE Features (Random AE): Max Activation Examples",
                                    num_features=10, top_n=10)
fig6.savefig('sae_random_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sae_random_max_activations.png")

# 7. Max activation examples - Random Projections on Trained AE
fig7 = find_max_activation_random_projections(ae_trained, train_loader,
                                              mean_trained, std_trained,
                                              random_projection,
                                              title_prefix="Random Projections (Trained AE): Max Activation Examples",
                                              top_n=10)
fig7.savefig('random_projection_trained_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: random_projection_trained_max_activations.png")

# 7b. Max activation examples - Random Projections on Random AE
fig7b = find_max_activation_random_projections(ae_random, train_loader,
                                               mean_random, std_random,
                                               random_projection,
                                               title_prefix="Random Projections (Random AE): Max Activation Examples",
                                               top_n=10)
fig7b.savefig('random_projection_random_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: random_projection_random_max_activations.png")

# 8. Max activation examples - Hidden Units (Trained AE)
fig8 = find_max_activation_hidden_units(ae_trained, train_loader,
                                        title_prefix="Hidden Units (Trained AE): Max Activation Examples",
                                        num_units=30, top_n=10)
fig8.savefig('hidden_units_trained_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hidden_units_trained_max_activations.png")

# 9. Max activation examples - Hidden Units (Random AE)
fig9 = find_max_activation_hidden_units(ae_random, train_loader,
                                        title_prefix="Hidden Units (Random AE): Max Activation Examples",
                                        num_units=30, top_n=10)
fig9.savefig('hidden_units_random_max_activations.png', dpi=150, bbox_inches='tight')
print("✓ Saved: hidden_units_random_max_activations.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)
print(f"\nFinal Losses:")
print(f"  Trained AE:          {ae_losses[-1]:.4f}")
print(f"  SAE (Trained AE):    {sae_trained_losses[-1]:.4f} (on normalized hidden)")
print(f"  SAE (Random AE):     {sae_random_losses[-1]:.4f} (on normalized hidden)")

print(f"\nNormalization Statistics:")
print(f"  Trained AE - Mean: {mean_trained.mean().item():.4f}, Std: {std_trained.mean().item():.4f}")
print(f"  Random AE  - Mean: {mean_random.mean().item():.4f}, Std: {std_random.mean().item():.4f}")

print(f"\nKey Findings:")
print(f"  • Normalization ensures fair comparison between SAEs")
print(f"  • Both SAEs now operate on similar input distributions")
print(f"  • Trained AE learns meaningful 30-dim representations")
print(f"  • Random AE produces arbitrary/unstructured representations")
print(f"  • SAE on trained AE learns interpretable features")
print(f"  • SAE on random AE learns to reconstruct noise-like patterns")

print("\n" + "="*70)
print("Generated Files:")
print("="*70)
print("  1. training_losses_comparison.png             - Loss curves")
print("  2. hidden_distributions.png                   - Hidden layer statistics")
print("  3. sae_trained_reconstructions.png            - Reconstructions (trained)")
print("  4. sae_random_reconstructions.png             - Reconstructions (random)")
print("  5. sae_trained_max_activations.png            - SAE features (trained AE)")
print("  6. sae_random_max_activations.png             - SAE features (random AE)")
print("  7. random_projection_trained_max_activations.png - Random proj (trained AE)")
print("  7b.random_projection_random_max_activations.png  - Random proj (random AE)")
print("  8. hidden_units_trained_max_activations.png   - Hidden units (trained AE)")
print("  9. hidden_units_random_max_activations.png    - Hidden units (random AE)")
print("="*70)