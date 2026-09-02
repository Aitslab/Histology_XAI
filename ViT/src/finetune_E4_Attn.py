import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import normalize
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import argparse
import cpuinfo
import platform
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== FIX: Disable LaTeX ====================
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

# ==================== CONSTANTS ====================
seed = 42
nfolds = 3
nclasses = 3
image_size = 224
batch_size = 16
num_epochs = 30
learning_rate = 1e-4
weight_decay = 1e-4

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# ==================== MAPPING ====================
mapping_labels = {
    "low": 0,
    "medium": 1,
    "high": 2
}

class_names = ["Low", "Medium", "High"]
mapping_reverse = {0: "low", 1: "medium", 2: "high"}

# ==================== EFFICIENTNET-B4 + ATTENTION MODEL ====================
class EfficientNetB4AttentionModel(nn.Module):
    """
    EfficientNet-B4 + Attention model.
    Matches Keras architecture: 64 -> 16 -> 1 with 1024 FC layer.
    EfficientNet-B4 feature map size: 7x7 with 1792 channels
    """
    def __init__(self, num_classes=3):
        super(EfficientNetB4AttentionModel, self).__init__()
        
        # ============================================================
        # 1. BACKBONE: EfficientNet-B4 (Frozen)
        # ============================================================
        self.backbone = timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=0)
        
        print("🔒 Freezing EfficientNet-B4 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get the actual feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone.forward_features(dummy_input)
            self.pt_depth = dummy_features.shape[1]  # This will be 1792
            print(f"✅ Detected feature depth: {self.pt_depth}")
        
        # ============================================================
        # 2. BATCH NORMALIZATION
        # ============================================================
        self.bn_features = nn.BatchNorm2d(self.pt_depth)
        
        # ============================================================
        # 3. ATTENTION LAYERS (matches Keras: 64 -> 16 -> 1)
        # ============================================================
        self.attn_conv1 = nn.Conv2d(self.pt_depth, 64, kernel_size=1, padding=0)
        self.attn_relu1 = nn.ReLU(inplace=True)
        
        self.attn_conv2 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.attn_relu2 = nn.ReLU(inplace=True)
        
        # LocallyConnected2D equivalent - using Conv2d with kernel_size=1
        # Note: PyTorch Conv2d with kernel_size=1 is equivalent to LocallyConnected2D
        self.attn_conv3 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.attn_sigmoid = nn.Sigmoid()
        
        # ============================================================
        # 4. UPSAMPLE ATTENTION (fan out to all channels)
        # ============================================================
        self.up_c2 = nn.Conv2d(1, self.pt_depth, kernel_size=1, padding=0, bias=False)
        with torch.no_grad():
            self.up_c2.weight.data.fill_(1.0)
        self.up_c2.weight.requires_grad = False
        
        # ============================================================
        # 5. CLASSIFICATION HEAD (matches Keras: 1024 FC, Dropout 0.5, 0.25)
        # ============================================================
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.pt_depth, 1024)
        self.elu = nn.ELU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, num_classes)
        
        print(f"✅ EfficientNet-B4 + Attention Model created with {num_classes} classes")
        print(f"   Backbone: EfficientNet-B4 (frozen)")
        print(f"   Feature depth: {self.pt_depth}")
        print(f"   Architecture: {self.pt_depth} -> 64 -> 16 -> 1 -> 1024 -> {num_classes}")
    
    def forward(self, x):
        """
        Full forward pass for classification.
        """
        # Backbone - EfficientNet-B4
        pt_features = self.backbone.forward_features(x)  # [B, 1792, 7, 7]
        
        # Batch Normalization
        bn_features = self.bn_features(pt_features)  # [B, 1792, 7, 7]
        
        # Attention mechanism (64 -> 16 -> 1)
        attn_layer = self.attn_conv1(bn_features)  # [B, 64, 7, 7]
        attn_layer = self.attn_relu1(attn_layer)
        
        attn_layer = self.attn_conv2(attn_layer)  # [B, 16, 7, 7]
        attn_layer = self.attn_relu2(attn_layer)
        
        attn_layer = self.attn_conv3(attn_layer)  # [B, 1, 7, 7]
        attn_layer = self.attn_sigmoid(attn_layer)
        
        # Store attention for later retrieval
        self.attn_output = attn_layer
        
        # Fan out attention to all channels (upsample)
        attn_layer_upsampled = self.up_c2(attn_layer)  # [B, pt_depth, 7, 7]
        
        # Apply attention mask (multiply)
        mask_features = attn_layer_upsampled * bn_features  # [B, pt_depth, 7, 7]
        
        # Global Average Pooling
        gap_features = self.gap(mask_features).squeeze(-1).squeeze(-1)  # [B, pt_depth]
        gap_mask = self.gap(attn_layer_upsampled).squeeze(-1).squeeze(-1)  # [B, pt_depth]
        
        # Rescale GAP (to account for missing values from attention)
        gap = gap_features / (gap_mask + 1e-8)
        
        # Classification Head
        gap_dr = self.dropout1(gap)  # [B, pt_depth]
        dr_steps = self.fc1(gap_dr)  # [B, 1024]
        dr_steps = self.elu(dr_steps)
        dr_steps = self.dropout2(dr_steps)  # [B, 1024]
        out_layer = self.fc2(dr_steps)  # [B, 3]
        
        return out_layer
    
    def get_attention_maps(self, x):
        """
        Extract attention maps.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        
        Returns:
            attention_maps: [batch_size, 1, 7, 7] - 7x7 attention maps
        """
        pt_features = self.backbone.forward_features(x)
        bn_features = self.bn_features(pt_features)
        
        attn_layer = self.attn_conv1(bn_features)
        attn_layer = self.attn_relu1(attn_layer)
        attn_layer = self.attn_conv2(attn_layer)
        attn_layer = self.attn_relu2(attn_layer)
        attn_layer = self.attn_conv3(attn_layer)
        attn_layer = self.attn_sigmoid(attn_layer)
        
        return attn_layer  # [B, 1, 7, 7]


# ==================== DATASET CLASS ====================
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== DATA TRANSFORMS ====================
def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


# ==================== UTILITY FUNCTIONS ====================
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def identify_device(seed):
    """Identifies the device (CPU or GPU) to be used for computations."""
    set_seed(seed)
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        try:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
        except:
            dev_name = "Apple Silicon"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
        else:
            try:
                dev_name = cpuinfo.get_cpu_info()["brand_raw"]
            except:
                dev_name = f"CPU ({platform.machine()})"
    return device, dev_name

def read_image(img_path):
    return np.array(Image.open(img_path).resize((image_size, image_size)))

def load_data():
    """Load data from folder structure with 3 classes."""
    images, labels = [], []
    
    path = "../dataset"
    print("Loading dataset.....")
    
    for folder in os.listdir(path):
        if folder not in mapping_labels:
            print(f"Warning: Folder '{folder}' not in mapping_labels. Skipping...")
            continue
            
        label = mapping_labels[folder]
        folder_path = os.path.join(path, folder)
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
        print(f"  Loading {len(image_files)} images from '{folder}' -> Class {label}")
        
        for imname in image_files:
            img_path = os.path.join(folder_path, imname)
            try:
                image = read_image(img_path)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    unique, counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({class_names[cls]}): {count} images")
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


# ==================== INFERENCE TIME MEASUREMENT ====================
def measure_inference_time(model, dataloader, device, num_batches=50):
    """
    Measure inference time for PyTorch model.
    """
    model.eval()
    
    # Warm-up runs
    print("🔥 Warming up...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= 10:
                break
            images = images.to(device)
            _ = model(images)
    
    # Measure inference time
    batch_times = []
    total_samples = 0
    
    print(f"⏱️ Measuring inference time over {num_batches} batches...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            images = images.to(device)
            batch_size = images.size(0)
            
            # Start timing
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            # Forward pass
            _ = model(images)
            
            # End timing
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_samples += batch_size
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    
    results = {
        'total_time': np.sum(batch_times),
        'mean_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times),
        'mean_per_sample': np.mean(batch_times) / dataloader.batch_size * 1000,  # ms per sample
        'throughput': total_samples / np.sum(batch_times),  # samples per second
        'num_samples': total_samples,
        'num_batches': len(batch_times)
    }
    
    return results


# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def train_fold(model, train_loader, val_loader, fold, device, args):
    """Train a single fold."""
    criterion = nn.CrossEntropyLoss()
    
    # Only attention and classification head trainable (backbone frozen)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    best_model_state = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✅ New best model! Val Acc: {best_val_acc:.4f}")
        
        print("-" * 60)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accs, val_accs, best_val_acc


# ==================== EVALUATION FUNCTIONS ====================
def evaluate_model(model, test_loader, device, fold, args):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n📊 Fold {fold + 1} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs(f"../experiments/{args.output_dir}/confusion_matrices/", exist_ok=True)
    plt.savefig(f"../experiments/{args.output_dir}/confusion_matrices/fold_{fold+1}_cm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return accuracy, all_preds, all_labels, all_probs


# ==================== PLOTTING FUNCTIONS ====================
def plot_training_history(train_losses, val_losses, train_accs, val_accs, fold, args):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves - Fold {fold + 1}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Curves - Fold {fold + 1}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(f"../experiments/{args.output_dir}/training_curves/", exist_ok=True)
    plt.savefig(f"../experiments/{args.output_dir}/training_curves/fold_{fold+1}_history.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==================== MAIN TRAINING FUNCTION ====================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B4 with Attention")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="../dataset",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="efficientnet_b4_attention",
                        help="Output directory name")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--folds", type=int, default=3,
                        help="Number of folds for cross-validation")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_save", action='store_true',
                        help="Don't save model checkpoints")
    
    args = parser.parse_args()
    
    # Set device
    device, devname = identify_device(args.seed)
    
    print(f"================================================================")
    print(f"Device: {device} - {devname}")
    print(f"Model: EfficientNet-B4 + Attention")
    print(f"Architecture: 1792 -> 64 -> 16 -> 1 -> 1024 -> {nclasses}")
    print(f"Backbone: Frozen (base_pretrained_model.trainable = False)")
    print(f"================================================================")
    print()
    
    # Load data
    images, labels = load_data()
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    
    fold_accuracies = []
    all_fold_preds = []
    all_fold_labels = []
    all_inference_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"{'='*60}")
        
        # Split data
        train_images, train_labels = images[train_idx], labels[train_idx]
        val_images, val_labels = images[val_idx], labels[val_idx]
        
        # Create datasets
        train_dataset = CustomDataset(train_images, train_labels, transform=train_transform)
        val_dataset = CustomDataset(val_images, val_labels, transform=val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # Create model
        model = EfficientNetB4AttentionModel(num_classes=nclasses).to(device)
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n📊 Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {frozen_params:,}")
        print(f"  Trainable Ratio: {trainable_params/total_params*100:.2f}%")
        
        # Measure inference time before training (for comparison)
        print("\n⏱️ Measuring inference time...")
        inference_results = measure_inference_time(model, val_loader, device, num_batches=50)
        all_inference_times.append(inference_results)
        
        print(f"\n📊 Inference Time (Pre-training):")
        print(f"  Throughput: {inference_results['throughput']:.2f} samples/sec")
        print(f"  Per sample: {inference_results['mean_per_sample']:.2f}ms")
        
        # Train fold
        model, train_losses, val_losses, train_accs, val_accs, best_val_acc = train_fold(
            model, train_loader, val_loader, fold, device, args
        )
        
        # Plot training history
        plot_training_history(train_losses, val_losses, train_accs, val_accs, fold, args)
        
        # Measure inference time after training
        print("\n⏱️ Measuring inference time after training...")
        inference_results_post = measure_inference_time(model, val_loader, device, num_batches=50)
        
        print(f"\n📊 Inference Time (Post-training):")
        print(f"  Throughput: {inference_results_post['throughput']:.2f} samples/sec")
        print(f"  Per sample: {inference_results_post['mean_per_sample']:.2f}ms")
        
        # Evaluate on validation set
        val_dataset_eval = CustomDataset(val_images, val_labels, transform=val_transform)
        val_loader_eval = DataLoader(val_dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        val_acc, preds, labels_true, probs = evaluate_model(model, val_loader_eval, device, fold, args)
        fold_accuracies.append(val_acc)
        all_fold_preds.extend(preds)
        all_fold_labels.extend(labels_true)
        
        # Save model
        if not args.no_save:
            os.makedirs(f"../experiments/{args.output_dir}/models/", exist_ok=True)
            torch.save({
                'fold': fold + 1,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'inference_time': inference_results_post,
                'trainable_params': trainable_params,
                'total_params': total_params,
            }, f"../experiments/{args.output_dir}/models/fold_{fold+1}_model.pt")
            print(f"✅ Model saved to ../experiments/{args.output_dir}/models/fold_{fold+1}_model.pt")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    
    # Inference time summary
    print(f"\n📊 Inference Time Summary:")
    mean_throughput = np.mean([t['throughput'] for t in all_inference_times])
    mean_per_sample = np.mean([t['mean_per_sample'] for t in all_inference_times])
    print(f"  Mean Throughput: {mean_throughput:.2f} samples/sec")
    print(f"  Mean Per Sample: {mean_per_sample:.2f}ms")
    
    # Overall classification report
    print(f"\nOverall Classification Report:")
    print(classification_report(all_fold_labels, all_fold_preds, target_names=class_names))
    
    # Overall confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_fold_labels, all_fold_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(f"../experiments/{args.output_dir}/", exist_ok=True)
    plt.savefig(f"../experiments/{args.output_dir}/overall_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    with open(f"../experiments/{args.output_dir}/training_summary.txt", 'w') as f:
        f.write(f"EfficientNet-B4 + Attention Training Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Device: {device} - {devname}\n")
        f.write(f"Folds: {args.folds}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.learning_rate}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Backbone: Frozen (base_pretrained_model.trainable = False)\n")
        f.write(f"\nModel Parameters:\n")
        f.write(f"  Total: {total_params:,}\n")
        f.write(f"  Trainable: {trainable_params:,}\n")
        f.write(f"  Frozen: {frozen_params:,}\n")
        f.write(f"  Trainable Ratio: {trainable_params/total_params*100:.2f}%\n")
        f.write(f"\nInference Time:\n")
        f.write(f"  Mean Throughput: {mean_throughput:.2f} samples/sec\n")
        f.write(f"  Mean Per Sample: {mean_per_sample:.2f}ms\n")
        f.write(f"\nFold Accuracies: {fold_accuracies}\n")
        f.write(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(all_fold_labels, all_fold_preds, target_names=class_names))
    
    print(f"\n✅ Training complete!")
    print(f"   Results saved to ../experiments/{args.output_dir}/")

if __name__ == "__main__":
    main()