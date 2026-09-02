import platform
import torch
import cpuinfo
import random 
import os 
import numpy as np
import argparse
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import timm
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

# ==================== CONFIGURATION ====================
seed = 42  # Changed for better reproducibility
nfolds = 3
img_size = (224, 224)
lr = 0.002
epochs = 30
mini_batch_size = 32
nclasses = 3  # Changed from 2 to 3
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# ==================== MAPPING ====================
# Define your class mappings - CHANGE THIS FOR YOUR DATA
mapping_labels = {
    "low": 0,  # e.g., "Low": 0
    "medium": 1,  # e.g., "Medium": 1
    "high": 2   # e.g., "High": 2
}

# For reporting
class_names = ["Low", "Medium", "High"]  # Change to your actual class names

# ==================== UTILITY FUNCTIONS ====================
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def identify_device():
    """
    Identifies the best available device (GPU > MPS > CPU).
    Returns device and device name.
    """
    so = platform.system()
    
    # Check for CUDA (NVIDIA GPU) - Windows/Linux
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
        set_seed(seed)
        print(f"✅ Using NVIDIA GPU: {dev_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device, dev_name
    
    # Check for MPS (Apple Silicon) - macOS
    elif so == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
        print(f"✅ Using Apple MPS (Metal Performance Shaders): {dev_name}")
        return device, dev_name
    
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
        print(f"⚠️  No GPU found. Using CPU: {dev_name}")
        if so != "Darwin":
            print("   Tip: Install CUDA version of PyTorch for GPU acceleration")
        return device, dev_name

    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, required=True, choices=["tiny", "small", "base"], 
                        help="ViT architecture")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("-b", "--batch", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    return args.type, args.epochs, args.batch

def read_image(img_path):
    return np.array(Image.open(img_path).resize(img_size))

def load_data():
    """Load data from folder structure with 3 classes."""
    images, labels = [], []
    
    path = "../dataset"
    print("Loading dataset.....")
    
    # Check if all mapping folders exist
    for folder in os.listdir(path):
        if folder not in mapping_labels:
            print(f"Warning: Folder '{folder}' not in mapping_labels. Skipping...")
            continue
            
        label = mapping_labels[folder]
        folder_path = os.path.join(path, folder)
        
        # Count images for logging
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
    
    # Print class distribution
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
    trainloader = DataLoader(data, shuffle=True, batch_size=mini_batch_size)
    
    data = list(zip(x_test, y_test))
    print(f"Test set size: {len(data)}")
    testloader = DataLoader(data, shuffle=False, batch_size=mini_batch_size)
    
    return trainloader, testloader

def load_vit(device, arch, num_classes=3):
    """Load ViT model with custom number of classes."""
    name = f"vit_{arch}_patch16_224"
    print(f"Loading {name} with {num_classes} classes...")
    
    # Load pretrained model (remove the last layer)
    model = timm.create_model(name, pretrained=True, num_classes=0)  # num_classes=0 removes classifier
    
    # Add custom classification head
    if arch == "tiny":
        hidden_dim = 192
    elif arch == "small":
        hidden_dim = 384
    else:  # base
        hidden_dim = 768
    
    # Create custom head
    model.head = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    # Option to freeze some layers (for transfer learning)
    # Uncomment to freeze backbone:
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.head.parameters():
    #     param.requires_grad = True
    
    model = model.to(device)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def train_vit(device, model, trainloader, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Reduced step size
    
    model = model.float()
    start = pc()
    
    print("Training..")
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
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
    
    t = pc() - start
    training_time = timedelta(seconds=t)
    print(f"Training completed in: {str(training_time)}")
    torch.cuda.empty_cache()
    
    return model, t

def save_model(model, arch, fold=1):
    path = f"../experiments/{arch}/"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/model_fold{fold}.pt")
    print(f"Model saved to {path}/model_fold{fold}.pt")

def predict(device, model, testloader):
    y_true, y_pred = [], []
    model.eval()
    start = pc()
    
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
    
    inference_time = pc() - start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    torch.cuda.empty_cache()
    
    return y_true, y_pred, inference_time

def evaluate_model(fold, target, predictions, inference_time):
    """Evaluate model with multi-class metrics."""
    acc = accuracy_score(target, predictions)
    
    # For multi-class (3 classes), use average='macro' for per-class average
    # Or use average='micro' for total accuracy
    # Or use average='weighted' for weighted average
    sens = recall_score(target, predictions, average='macro')  # Macro average
    spec = recall_score(target, predictions, average='macro')  # For multi-class, spec is same as sens
    
    # For per-class metrics
    per_class_recall = recall_score(target, predictions, average=None)
    f1 = f1_score(target, predictions, average='macro')
    
    print(f"Fold {fold} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro Recall: {sens:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    print(f"  Per-class Recall: {[f'{r:.4f}' for r in per_class_recall]}")
    print(f"  Inference Time: {inference_time:.4f}s")
    
    return [fold, acc, sens, spec, f1, inference_time]

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

def save_report(results, arch):
    columns = ["Fold", "Accuracy", "Macro_Recall", "Macro_Specificity", "Macro_F1", "Training_Time(s)"]
    path = f"../experiments/{arch}/metrics.csv"
    os.makedirs(f"../experiments/{arch}/", exist_ok=True)
    
    metrics = pd.DataFrame(results, columns=columns)
    metrics.to_csv(path, index=False)
    print(f"\nMetrics saved to {path}")
    
    # Calculate averages
    data = {}
    print(f"\nAverage results across {len(results)} folds:")
    for column in metrics.columns[1:]:
        values = metrics[column]
        mu, sigma = np.mean(values), np.std(values)
        print(f"  {column}: {mu:.4f} ± {sigma:.4f}")
        data[column] = {
            "Mean": float(mu),
            "Standard Deviation": float(sigma)
        }
    
    # Save summary
    path = f"../experiments/{arch}/results.yaml"
    with open(path, "w") as file:
        yaml.dump(data, file)
    print(f"Summary saved to {path}")

def run_exp(device, arch, epochs=30):
    """Run the complete experiment."""
    images, labels = load_data()
    report = []
    
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    for fold, (train_index, test_index) in enumerate(skf.split(images, labels), start=1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{nfolds}")
        print(f"{'='*60}")
        
        trainloader, testloader = create_train_test_sets(train_index, test_index, images, labels)
        model = load_vit(device, arch, num_classes=nclasses)
        model, train_time = train_vit(device, model, trainloader, epochs=epochs)
        
        target, predictions, inference_time = predict(device, model, testloader)
        
        # Compute confusion matrix
        conf_mat = confusion_matrix(target, predictions)
        plot_confusion_matrix(conf_mat, fold, arch)
        
        rep = evaluate_model(fold, target, predictions, inference_time)
        rep.append(train_time)
        report.append(rep)
        
        # Save model only for fold 1 (or all folds if you want)
        save_model(model, arch, fold=fold)
        
        print(f"{'='*60}\n")
    
    save_report(report, arch)

def main():

    
    # Update global batch size
    global mini_batch_size, epochs

    device, dev_name = identify_device()
    arch, epochs, batch = parse_arguments()
    
    mini_batch_size = batch
    epochs = epochs
    
    print(f"{'='*60}")
    print(f"Device: {device} - {dev_name}")
    print(f"ViT Architecture: {arch}")
    print(f"Number of Classes: {nclasses}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {mini_batch_size}")
    print(f"{'='*60}\n")
    
    run_exp(device, arch, epochs)
    
    print("\n✅ Experiment completed successfully!")
    exit(0)

if __name__ == "__main__":
    main()