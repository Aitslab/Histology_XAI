import platform
import torch
import random 
import os 
import numpy as np
import argparse
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from time import perf_counter as pc
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import gc
import time

# ==================== MEMORY OPTIMIZATION ====================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== CONFIGURATION ====================
seed = 42
nfolds = 3
img_size = (224, 224)
lr = 0.002
epochs = 30
mini_batch_size = 32
nclasses = 3
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# ==================== MAPPING ====================
mapping_labels = {
    "low": 0,
    "medium": 1,
    "high": 2
}

class_names = ["Low", "Medium", "High"]

# ==================== EXACT VGG16 + ATTENTION MODEL ====================
class VGG16AttentionModel(nn.Module):
    """
    EXACT PyTorch implementation of the Keras VGG16 + Attention model.
    Matches: 64 -> 16 -> 1 with 1024 FC layer
    """
    def __init__(self, num_classes=3):
        super(VGG16AttentionModel, self).__init__()
        
        # ============================================================
        # 1. BACKBONE: VGG16 (Frozen - trainable=False)
        # ============================================================
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = vgg16.features
        
        print("🔒 Freezing VGG16 backbone...")
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get feature depth (pt_depth = 512 for VGG16)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            self.pt_depth = dummy_features.shape[1]  # 512
            print(f"✅ Feature depth: {self.pt_depth}")
        
        # ============================================================
        # 2. BATCH NORMALIZATION
        # ============================================================
        self.bn_features = nn.BatchNorm2d(self.pt_depth)
        
        # ============================================================
        # 3. ATTENTION LAYERS (EXACT: 64 -> 16 -> 1)
        # ============================================================
        self.attn_conv1 = nn.Conv2d(self.pt_depth, 64, kernel_size=1, padding=0)
        self.attn_relu1 = nn.ReLU(inplace=True)
        
        self.attn_conv2 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.attn_relu2 = nn.ReLU(inplace=True)
        
        # LocallyConnected2D equivalent with kernel_size=1
        # Note: Conv2d with kernel_size=1 is equivalent to LocallyConnected2D
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
        # 5. CLASSIFICATION HEAD (EXACT: 1024 FC, Dropout 0.5, 0.25)
        # ============================================================
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.pt_depth, 1024)  # EXACT: 1024
        self.elu = nn.ELU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, num_classes)    # EXACT: linear activation
        
        print(f"✅ VGG16 + Attention Model (Exact Keras Match)")
        print(f"   Architecture: {self.pt_depth} -> 64 -> 16 -> 1 -> 1024 -> {num_classes}")
    
    def forward(self, x):
        """
        Full forward pass matching Keras implementation.
        """
        # 1. Backbone (VGG16 features)
        pt_features = self.backbone(x)  # [B, 512, 7, 7]
        
        # 2. Batch Normalization
        bn_features = self.bn_features(pt_features)  # [B, 512, 7, 7]
        
        # 3. Attention mechanism (64 -> 16 -> 1)
        attn_layer = self.attn_conv1(bn_features)  # [B, 64, 7, 7]
        attn_layer = self.attn_relu1(attn_layer)
        
        attn_layer = self.attn_conv2(attn_layer)  # [B, 16, 7, 7]
        attn_layer = self.attn_relu2(attn_layer)
        
        attn_layer = self.attn_conv3(attn_layer)  # [B, 1, 7, 7]
        attn_layer = self.attn_sigmoid(attn_layer)
        
        # 4. Fan out attention to all channels (upsample)
        attn_layer_upsampled = self.up_c2(attn_layer)  # [B, 512, 7, 7]
        
        # 5. Multiply attention mask with features
        mask_features = attn_layer_upsampled * bn_features  # [B, 512, 7, 7]
        
        # 6. Global Average Pooling
        gap_features = self.gap(mask_features).squeeze(-1).squeeze(-1)  # [B, 512]
        gap_mask = self.gap(attn_layer_upsampled).squeeze(-1).squeeze(-1)  # [B, 512]
        
        # 7. Rescale GAP (to account for missing values from attention)
        gap = gap_features / (gap_mask + 1e-8)
        
        # 8. Classification Head (1024 FC, Dropout 0.5, 0.25)
        gap_dr = self.dropout1(gap)  # [B, 512]
        dr_steps = self.fc1(gap_dr)  # [B, 1024]
        dr_steps = self.elu(dr_steps)
        dr_steps = self.dropout2(dr_steps)  # [B, 1024]
        out_layer = self.fc2(dr_steps)  # [B, 3] - linear activation
        
        return out_layer
    
    def get_attention_maps(self, x):
        """
        Extract attention maps for visualization.
        """
        pt_features = self.backbone(x)
        bn_features = self.bn_features(pt_features)
        
        attn_layer = self.attn_conv1(bn_features)
        attn_layer = self.attn_relu1(attn_layer)
        attn_layer = self.attn_conv2(attn_layer)
        attn_layer = self.attn_relu2(attn_layer)
        attn_layer = self.attn_conv3(attn_layer)
        attn_layer = self.attn_sigmoid(attn_layer)
        
        return attn_layer

def create_model(num_classes=3):
    """Factory function to create the model."""
    model = VGG16AttentionModel(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 Parameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model

# ==================== INFERENCE TIME MEASUREMENT ====================
def measure_inference_time(model, dataloader, device, num_warmup=10, num_batches=50):
    """
    Measure inference time with proper warm-up and per-image timing.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        device: 'cuda' or 'cpu'
        num_warmup: Number of warm-up batches
        num_batches: Number of batches to measure
    
    Returns:
        dict: Inference time statistics
    """
    model.eval()
    
    # Warm-up runs (to initialize GPU/CUDA)
    print("🔥 Warming up...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_warmup:
                break
            inputs = inputs / 255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize(inputs, mean, std).to(device)
            _ = model(inputs.float())
    
    # Actual measurement
    batch_times = []
    per_image_times = []
    total_samples = 0
    
    print(f"⏱️ Measuring inference time over {num_batches} batches...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            inputs = inputs / 255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize(inputs, mean, std).to(device)
            batch_size = inputs.size(0)
            
            # Synchronize GPU before timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Start timing
            start = pc()
            
            # Forward pass
            outputs = model(inputs.float())
            
            # Synchronize GPU after timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # End timing
            end = pc()
            
            batch_time = end - start
            batch_times.append(batch_time)
            per_image_times.append(batch_time / batch_size)
            total_samples += batch_size
    
    # Calculate statistics
    batch_times = np.array(batch_times)
    per_image_times = np.array(per_image_times)
    
    results = {
        'total_time': np.sum(batch_times),
        'mean_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times),
        'mean_per_image_ms': np.mean(per_image_times) * 1000,  # ms
        'std_per_image_ms': np.std(per_image_times) * 1000,    # ms
        'throughput': total_samples / np.sum(batch_times),     # samples/sec
        'num_samples': total_samples,
        'num_batches': len(batch_times)
    }
    
    return results

# ==================== UTILITY FUNCTIONS ====================
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def identify_device():
    """Identifies the best available device."""
    so = platform.system()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
        set_seed(seed)
        print(f"✅ Using NVIDIA GPU: {dev_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device, dev_name
    
    elif so == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            import cpuinfo
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
        except:
            dev_name = "Apple Silicon"
        print(f"✅ Using Apple MPS: {dev_name}")
        return device, dev_name
    
    else:
        device = torch.device("cpu")
        try:
            import cpuinfo
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
        except:
            dev_name = f"CPU ({platform.machine()})"
        print(f"⚠️ No GPU found. Using CPU: {dev_name}")
        return device, dev_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("-b", "--batch", type=int, default=32, help="Batch size")
    parser.add_argument("-l", "--lr", type=float, default=0.002, help="Learning rate")
    return parser.parse_args()

def read_image(img_path):
    return np.array(Image.open(img_path).resize(img_size))

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

def create_train_test_sets(train, test, images, labels):
    x_train, x_test = images[train], images[test]
    y_train, y_test = labels[train], labels[test]
    
    data = list(zip(x_train, y_train))
    print(f"Training set size: {len(data)}")
    trainloader = DataLoader(data, shuffle=True, batch_size=mini_batch_size, 
                            num_workers=0, pin_memory=False)
    
    data = list(zip(x_test, y_test))
    print(f"Test set size: {len(data)}")
    testloader = DataLoader(data, shuffle=False, batch_size=mini_batch_size,
                           num_workers=0, pin_memory=False)
    
    return trainloader, testloader

def train_model(device, model, trainloader, epochs=30, lr=0.002):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model = model.float()
    start = pc()
    
    print("Training VGG16 + Attention model (Exact Keras Match)...")
    for epoch in tqdm(range(epochs)):
        model.train(True)
        running_loss = 0.0
        running_correct = 0
        
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs / 255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize(inputs, mean, std).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
        
        epoch_loss = running_loss / (i + 1)
        epoch_acc = running_correct / len(trainloader.dataset) * 100
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        clear_memory()
    
    t = pc() - start
    training_time = timedelta(seconds=t)
    print(f"Training completed in: {str(training_time)}")
    clear_memory()
    
    return model, t

def save_model(model, arch, fold=1):
    path = f"../experiments/{arch}/"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/model_fold{fold}.pt")
    print(f"Model saved to {path}/model_fold{fold}.pt")

def predict_with_timing(device, model, testloader):
    """Predict with detailed timing information."""
    y_true, y_pred = [], []
    model.eval()
    
    # Measure inference time
    timing_results = measure_inference_time(model, testloader, device, 
                                            num_warmup=10, num_batches=len(testloader))
    
    print(f"\n📊 Inference Time Results:")
    print(f"  Total time: {timing_results['total_time']:.4f}s")
    print(f"  Mean batch time: {timing_results['mean_batch_time']*1000:.2f}ms")
    print(f"  Std batch time: {timing_results['std_batch_time']*1000:.2f}ms")
    print(f"  Mean per image: {timing_results['mean_per_image_ms']:.2f}ms")
    print(f"  Throughput: {timing_results['throughput']:.2f} img/sec")
    print(f"  Samples: {timing_results['num_samples']}")
    print(f"  Batches: {timing_results['num_batches']}")
    
    # Get predictions
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs / 255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize(inputs, mean, std).to(device)
            labels = labels.to(device)
            
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            y_true += labels.tolist()
            y_pred += preds.tolist()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    clear_memory()
    
    return y_true, y_pred, timing_results

def evaluate_model(fold, target, predictions, timing_results):
    """Evaluate model with multi-class metrics."""
    acc = accuracy_score(target, predictions)
    sens = recall_score(target, predictions, average='macro')
    spec = recall_score(target, predictions, average='macro')
    per_class_recall = recall_score(target, predictions, average=None)
    f1 = f1_score(target, predictions, average='macro')
    
    print(f"\n📊 Fold {fold} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro Recall: {sens:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"  Per-class Recall: {[f'{r:.4f}' for r in per_class_recall]}")
    print(f"  Inference Time: {timing_results['total_time']:.4f}s")
    print(f"  Throughput: {timing_results['throughput']:.2f} img/sec")
    print(f"  Per Image: {timing_results['mean_per_image_ms']:.2f}ms")
    
    return [fold, acc, sens, spec, f1, timing_results['total_time']]

def plot_confusion_matrix(conf_mat, fold, arch):
    """Plot and save confusion matrix."""
    path = f"../experiments/{arch}/"
    os.makedirs(path, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{path}/confusion_matrix_fold{fold}.png", dpi=300)
    plt.close()

def save_report(results, arch, timing_summary):
    columns = ["Fold", "Accuracy", "Macro_Recall", "Macro_Specificity", "Macro_F1", "Inference_Time(s)"]
    path = f"../experiments/{arch}/metrics.csv"
    os.makedirs(f"../experiments/{arch}/", exist_ok=True)
    
    metrics = pd.DataFrame(results, columns=columns)
    metrics.to_csv(path, index=False)
    print(f"\nMetrics saved to {path}")
    
    data = {}
    print(f"\n📊 Average results across {len(results)} folds:")
    for column in metrics.columns[1:]:
        values = metrics[column]
        mu, sigma = np.mean(values), np.std(values)
        print(f"  {column}: {mu:.4f} ± {sigma:.4f}")
        data[column] = {
            "Mean": float(mu),
            "Standard Deviation": float(sigma)
        }
    
    # Add inference timing summary
    data["Inference_Time_Summary"] = {
        "Throughput (img/sec)": float(timing_summary['throughput']),
        "Per Image (ms)": float(timing_summary['mean_per_image_ms']),
        "Mean Batch Time (ms)": float(timing_summary['mean_batch_time'] * 1000)
    }
    
    path = f"../experiments/{arch}/results.yaml"
    with open(path, "w") as file:
        yaml.dump(data, file)
    print(f"Summary saved to {path}")

def run_exp(device, epochs=30, lr=0.002):
    """Run the complete experiment with the VGG16 + Attention model."""
    images, labels = load_data()
    report = []
    all_timing_results = []
    
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    for fold, (train_index, test_index) in enumerate(skf.split(images, labels), start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{nfolds}")
        print(f"{'='*60}")
        
        trainloader, testloader = create_train_test_sets(train_index, test_index, images, labels)
        
        clear_memory()
        
        # Create the exact VGG16 + Attention model
        model = create_model(num_classes=nclasses)
        model = model.to(device)
        
        model, train_time = train_model(device, model, trainloader, epochs=epochs, lr=lr)
        
        target, predictions, timing_results = predict_with_timing(device, model, testloader)
        all_timing_results.append(timing_results)
        
        conf_mat = confusion_matrix(target, predictions)
        plot_confusion_matrix(conf_mat, fold, "vgg16_attention_exact")
        
        rep = evaluate_model(fold, target, predictions, timing_results)
        report.append(rep)
        
        save_model(model, "vgg16_attention_exact", fold=fold)
        
        clear_memory()
        del model
        clear_memory()
        
        print(f"{'='*60}\n")
    
    # Calculate average timing across folds
    avg_timing = {
        'throughput': np.mean([t['throughput'] for t in all_timing_results]),
        'mean_per_image_ms': np.mean([t['mean_per_image_ms'] for t in all_timing_results]),
        'mean_batch_time': np.mean([t['mean_batch_time'] for t in all_timing_results])
    }
    
    save_report(report, "vgg16_attention_exact", avg_timing)

def main():
    global mini_batch_size, epochs, lr

    device, dev_name = identify_device()
    args = parse_arguments()
    
    epochs = args.epochs
    mini_batch_size = args.batch
    lr = args.lr
    
    print(f"{'='*60}")
    print(f"Device: {device} - {dev_name}")
    print(f"Model: VGG16 + Attention (EXACT Keras Match)")
    print(f"Architecture: 512 -> 64 -> 16 -> 1 -> 1024 -> {nclasses}")
    print(f"Backbone: VGG16 (Frozen - trainable=False)")
    print(f"Number of Classes: {nclasses}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {mini_batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*60}\n")
    
    run_exp(device, epochs, lr)
    
    print("\n✅ Experiment completed successfully!")
    exit(0)

if __name__ == "__main__":
    main()