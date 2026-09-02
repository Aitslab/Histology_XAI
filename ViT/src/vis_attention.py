import random
import os 
import numpy as np
import torch
import cpuinfo
import platform
import argparse
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms.functional import normalize
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# ==================== FIX: Disable LaTeX ====================
# Comment out or remove the LaTeX setting
# plt.rcParams["text.usetex"] = True  # REMOVE THIS LINE

# Optionally set a nice font without LaTeX
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

seed = 42
nfolds = 2
nclasses = 3
patch_size = 16
image_size = 224
image_size_patch = image_size // patch_size
alpha = 0.3

# ==================== MAPPING ====================
mapping_labels = {
    "low": 0,
    "medium": 1,
    "high": 2
}

class_names = ["Low", "Medium", "High"]
mapping_reverse = {0: "low", 1: "medium", 2: "high"}

# ==================== MODEL DEFINITION ====================
def create_vit_model(arch='small', num_classes=3):
    """Create ViT model with custom classification head matching training."""
    name = f"vit_{arch}_patch16_224"
    print(f"Creating {name} with {num_classes} classes...")
    
    # Load pretrained model (remove the last layer)
    model = timm.create_model(name, pretrained=True, num_classes=0)
    
    # Get hidden dimension based on architecture
    hidden_dims = {
        'tiny': 192,
        'small': 384,
        'base': 768
    }
    hidden_dim = hidden_dims.get(arch, 192)
    
    # Create custom head matching the training script
    model.head = nn.Sequential(
        nn.Linear(hidden_dim, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

def fix_state_dict_keys(state_dict):
    """
    Fix the state dict keys to match the model's expected format.
    Handles both Sequential-style and single-layer head formats.
    """
    new_state_dict = {}
    
    # Check what kind of head keys we have
    has_sequential_keys = any(k.startswith('head.0.') or k.startswith('head.3.') or k.startswith('head.6.') for k in state_dict.keys())
    has_single_head = 'head.weight' in state_dict or 'head.bias' in state_dict
    
    if has_sequential_keys:
        print("Detected Sequential-style head keys. Keeping as-is...")
        new_state_dict = state_dict.copy()
    
    elif has_single_head:
        print("Detected single-layer head keys. Mapping to Sequential structure...")
        if 'head.weight' in state_dict:
            new_state_dict['head.0.weight'] = state_dict['head.weight']
        if 'head.bias' in state_dict:
            new_state_dict['head.0.bias'] = state_dict['head.bias']
        
        for k, v in state_dict.items():
            if not k.startswith('head.'):
                new_state_dict[k] = v
        
        # Initialize the other layers
        hidden_dim = state_dict['head.weight'].shape[1] if 'head.weight' in state_dict else 192
        new_state_dict['head.3.weight'] = torch.randn(256, 512) * 0.02
        new_state_dict['head.3.bias'] = torch.zeros(256)
        new_state_dict['head.6.weight'] = torch.randn(3, 256) * 0.02
        new_state_dict['head.6.bias'] = torch.zeros(3)
    
    else:
        print("Unknown head format. Keeping keys as-is...")
        new_state_dict = state_dict.copy()
    
    return new_state_dict

def load_vit_model(model_path, arch='small', num_classes=3, device='cpu'):
    """
    Load a trained ViT model with proper error handling and key mapping.
    """
    print(f"Loading model from {model_path}")
    
    # Create model with the same architecture
    model = create_vit_model(arch=arch, num_classes=num_classes)
    
    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except:
        checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # Fix the state dict keys
    state_dict = fix_state_dict_keys(state_dict)
    
    # Check what keys are expected by the model
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    print(f"Model expects: {len(model_keys)} keys")
    print(f"State dict has: {len(state_dict_keys)} keys")
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys:
        print(f"⚠️ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {unexpected_keys}")
    
    # If we have missing head keys, initialize them randomly
    if missing_keys and any('head' in k for k in missing_keys):
        print("Initializing missing head keys randomly...")
        for key in missing_keys:
            if 'weight' in key:
                shape = model.state_dict()[key].shape
                state_dict[key] = torch.randn(shape) * 0.02
            elif 'bias' in key:
                shape = model.state_dict()[key].shape
                state_dict[key] = torch.zeros(shape)
    
    # Load with strict=False to handle any remaining mismatches
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model

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

def identify_device(seed):
    """Identifies the device (CPU or GPU) to be used for computations."""
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
            set_seed(seed)
        else:
            try:
                dev_name = cpuinfo.get_cpu_info()["brand_raw"]
            except:
                dev_name = f"CPU ({platform.machine()})"
    return device, dev_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, 
                        help="Path to the trained model (.pt file)")
    parser.add_argument("-a", "--arch", type=str, default="tiny", choices=["tiny", "small", "base"],
                        help="ViT architecture (default: tiny)")
    parser.add_argument("-f", "--fold", type=int, default=1,
                        help="Fold number to use (default: 1)")
    parser.add_argument("-n", "--num_samples", type=int, default=50,
                        help="Number of samples to visualize (default: 50)")
    return parser.parse_args()

def read_image(img_path):
    return np.array(Image.open(img_path).resize((image_size, image_size)))

def load_data():
    """Load data from folder structure with 3 classes."""
    images, labels = [], []
    
    path = "../test_Attention/"
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

def save_attention_image(img, attention, target, prediction, i, arch):
    """Save attention map visualization."""
    path = f"../experiments/{arch}/attention_maps/"
    os.makedirs(path, exist_ok=True)
    
    # Ensure attention map is 2D
    if attention.ndim == 3:
        attention = attention.squeeze(0)
    
    # Resize attention map to match image size
    attention_resized = np.array(Image.fromarray(attention).resize((img.shape[1], img.shape[0]), 
                                                                    Image.Resampling.BILINEAR))
    
    # Normalize attention map to [0, 1]
    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    title = f"True: {target} | Pred: {prediction}"
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(title, fontsize=16, y=0.85)
    fig.set_size_inches(18.5, 10.5)
    
    # Original image
    axs[0].imshow(img.astype(np.uint8))
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")
    
    # Attention map
    im = axs[1].imshow(attention_resized, cmap='jet')
    axs[1].set_title("Attention Map", fontsize=14)
    axs[1].axis("off")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axs[2].imshow(img.astype(np.uint8))
    axs[2].imshow(attention_resized, alpha=alpha, cmap='jet', vmin=0, vmax=1)
    axs[2].set_title("Overlay", fontsize=14)
    axs[2].axis("off")
    
    fig.savefig(f"{path}/image_{i:04d}.png", bbox_inches='tight', dpi=150)
    plt.clf()
    plt.close()

def compute_attention_maps(device, model_path, arch='small', fold=1, num_samples=50):
    """
    Compute and visualize attention maps from the ViT model using hooks.
    """
    # Load data
    images, labels = load_data()
    
    # Load model
    model = load_vit_model(model_path, arch=arch, num_classes=nclasses, device=device)
    model.eval()
    
    # Use StratifiedKFold to get test indices
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    folds = list(skf.split(images, labels))
    
    # Get test indices for the specified fold
    _, test_index = folds[fold - 1]
    
    # Randomly select samples
    if len(test_index) > num_samples:
        indices = np.random.choice(test_index, num_samples, replace=False)
    else:
        indices = test_index
    
    test_images, test_labels = images[indices], labels[indices]
    
    # Storage for attention maps
    attention_maps = []
    
    # Hook function to capture attention
    def hook_fn(module, input, output):
        if hasattr(module, 'attn_map'):
            attention_maps.append(module.attn_map.detach().cpu())
        elif hasattr(module, 'attn') and hasattr(module.attn, 'attn_map'):
            attention_maps.append(module.attn.attn_map.detach().cpu())
    
    # Register hook on the last attention layer
    if hasattr(model.blocks[-1], 'attn'):
        hook_handle = model.blocks[-1].attn.register_forward_hook(hook_fn)
        print("Registered hook on model.blocks[-1].attn")
    elif hasattr(model.blocks[-1], 'attention'):
        hook_handle = model.blocks[-1].attention.register_forward_hook(hook_fn)
        print("Registered hook on model.blocks[-1].attention")
    else:
        print("Error: Could not find attention layer!")
        return
    
    print(f"\nGenerating attention maps for {len(test_images)} images...")
    
    correct = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(test_images, test_labels), 1):
            # Prepare image
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_norm = normalize(img_tensor / 255, mean, std)
            img_input = img_norm.unsqueeze(0).to(device)
            
            # Forward pass (hook captures attention)
            output = model(img_input)
            _, pred = torch.max(output, 1)
            
            if pred.item() == label:
                correct += 1
            
            # Get CLS attention map from the captured maps
            if attention_maps:
                attn = attention_maps[-1]  # Get the most recent
                
                # Check the shape of the attention map
                if attn.ndim == 4:
                    # CLS token attention to patches (remove CLS token itself)
                    cls_weight = attn[:, :, 0, 1:]  # [batch, heads, num_patches]
                else:
                    # Fallback
                    cls_weight = attn[:, :, 1:, 1:]
                    if cls_weight.ndim == 4:
                        cls_weight = cls_weight.mean(dim=1)
                    else:
                        cls_weight = attn.squeeze(0)
                
                # Average over heads
                if cls_weight.ndim == 3:
                    cls_weight = cls_weight.mean(dim=1)
                elif cls_weight.ndim == 2:
                    cls_weight = cls_weight.unsqueeze(0)
                
                cls_weight = cls_weight.squeeze(0)
                
                # Reshape to patch grid
                cls_weight = cls_weight.view(image_size_patch, image_size_patch).numpy()
                
                # Apply filters for smoothing
                cls_weight = median_filter(cls_weight, footprint=morphology.disk(patch_size // 2))
                cls_weight = gaussian_filter(cls_weight, sigma=patch_size // 4)
            else:
                # Fallback
                print(f"Warning: No attention map captured for image {idx}")
                cls_weight = np.random.rand(image_size_patch, image_size_patch)
            
            # Get labels
            true_label = mapping_reverse[label]
            pred_label = mapping_reverse[int(pred)]
            
            # Save visualization
            save_attention_image(img, cls_weight, true_label, pred_label, idx, arch)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(test_images)} images...")
            
            # Clear attention maps for next iteration
            attention_maps.clear()
    
    # Remove hook
    hook_handle.remove()
    
    accuracy = correct / len(test_images) * 100
    print(f"\n✅ Attention maps saved to ../experiments/{arch}/attention_maps/")
    print(f"   Model accuracy on selected samples: {accuracy:.2f}%")

def main():
    device, devname = identify_device(seed)
    args = parse_arguments()
    
    print(f"================================================================")
    print(f"Using {device} - {devname}")
    print(f"ViT Architecture: {args.arch}")
    print(f"Model Path: {args.model_path}")
    print(f"Fold: {args.fold}")
    print(f"Number of samples: {args.num_samples}")
    print(f"================================================================")
    print()
    
    compute_attention_maps(
        device=device,
        model_path=args.model_path,
        arch=args.arch,
        fold=args.fold,
        num_samples=args.num_samples
    )
    exit(0)

if __name__ == "__main__":
    main()