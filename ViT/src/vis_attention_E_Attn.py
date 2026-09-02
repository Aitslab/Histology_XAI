import random
import os 
import numpy as np
import torch
import torch.nn as nn
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
from torchvision import models
import cv2
from matplotlib.patches import Rectangle

# ==================== FIX: Disable LaTeX ====================
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

seed = 42
nfolds = 5
nclasses = 3
patch_size = 16
image_size = 224
image_size_patch = image_size // patch_size
alpha = 0.8

# ==================== MAPPING ====================
mapping_labels = {
    "low": 0,
    "medium": 1,
    "high": 2
}

class_names = ["Low", "Medium", "High"]
mapping_reverse = {0: "low", 1: "medium", 2: "high"}

# ==================== EFFICIENTNET-B3 + ATTENTION MODEL ====================
class EfficientNetB3AttentionModel(nn.Module):
    """
    EfficientNet-B3 + Attention model.
    Matches the training script architecture.
    EfficientNet-B3: 1536 channels, 7x7 feature map
    """
    def __init__(self, num_classes=3, freeze_backbone=True):
        super(EfficientNetB3AttentionModel, self).__init__()
        
        # ============================================================
        # 1. BACKBONE: EfficientNet-B3
        # ============================================================
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        if freeze_backbone:
            print("🔒 Freezing EfficientNet-B3 backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔓 EfficientNet-B3 backbone is trainable...")
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        # Get the actual feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone.forward_features(dummy_input)
            self.pt_depth = dummy_features.shape[1]  # This will be 1536
            print(f"✅ Detected feature depth: {self.pt_depth}")
        
        # ============================================================
        # 2. BATCH NORMALIZATION
        # ============================================================
        self.bn_features = nn.BatchNorm2d(self.pt_depth)
        
        # ============================================================
        # 3. ATTENTION LAYERS
        # ============================================================
        self.attn_conv1 = nn.Conv2d(self.pt_depth, 32, kernel_size=1, padding=0)
        self.attn_relu1 = nn.ReLU(inplace=True)
        
        self.attn_conv2 = nn.Conv2d(32, 8, kernel_size=1, padding=0)
        self.attn_relu2 = nn.ReLU(inplace=True)
        
        self.attn_conv3 = nn.Conv2d(8, 1, kernel_size=1, padding=0)
        self.attn_sigmoid = nn.Sigmoid()
        
        # ============================================================
        # 4. UPSAMPLE ATTENTION
        # ============================================================
        self.up_c2 = nn.Conv2d(1, self.pt_depth, kernel_size=1, padding=0, bias=False)
        with torch.no_grad():
            self.up_c2.weight.data.fill_(1.0)
        self.up_c2.weight.requires_grad = False
        
        # ============================================================
        # 5. CLASSIFICATION HEAD
        # ============================================================
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.pt_depth, 512)
        self.elu = nn.ELU(inplace=True)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, num_classes)
        
        print(f"✅ EfficientNet-B3 + Attention Model created with {num_classes} classes")
        print(f"   Backbone: EfficientNet-B3 ({'frozen' if freeze_backbone else 'trainable'})")
        print(f"   Feature depth: {self.pt_depth}")
        print(f"   Architecture: {self.pt_depth} -> 32 -> 8 -> 1 -> 512 -> {num_classes}")
    
    def forward(self, x):
        """
        Full forward pass for classification.
        """
        # Backbone - EfficientNet-B3
        pt_features = self.backbone.forward_features(x)  # [B, 1536, 7, 7]
        
        # Batch Normalization
        bn_features = self.bn_features(pt_features)  # [B, 1536, 7, 7]
        
        # Attention mechanism
        attn_layer = self.attn_conv1(bn_features)  # [B, 32, 7, 7]
        attn_layer = self.attn_relu1(attn_layer)
        
        attn_layer = self.attn_conv2(attn_layer)  # [B, 8, 7, 7]
        attn_layer = self.attn_relu2(attn_layer)
        
        attn_layer = self.attn_conv3(attn_layer)  # [B, 1, 7, 7]
        attn_layer = self.attn_sigmoid(attn_layer)
        
        # Store attention for later retrieval
        self.attn_output = attn_layer
        
        # Upsample attention
        attn_layer_upsampled = self.up_c2(attn_layer)  # [B, pt_depth, 7, 7]
        
        # Apply attention mask
        mask_features = attn_layer_upsampled * bn_features  # [B, pt_depth, 7, 7]
        
        # Global Average Pooling
        gap_features = self.gap(mask_features).squeeze(-1).squeeze(-1)  # [B, pt_depth]
        gap_mask = self.gap(attn_layer_upsampled).squeeze(-1).squeeze(-1)  # [B, pt_depth]
        
        # Rescale GAP
        gap = gap_features / (gap_mask + 1e-8)
        
        # Classification Head
        gap_dr = self.dropout1(gap)  # [B, pt_depth]
        dr_steps = self.fc1(gap_dr)  # [B, 512]
        dr_steps = self.elu(dr_steps)
        dr_steps = self.dropout2(dr_steps)  # [B, 512]
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


# ==================== ATTENTION EXTRACTOR CLASS ====================
class AttentionExtractor:
    """
    Class to extract attention maps from EfficientNet-B3+Attention model.
    """
    def __init__(self, model_path, num_classes=3, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load the trained EfficientNet-B3+Attention model."""
        # Create model with frozen backbone (matches training)
        model = EfficientNetB3AttentionModel(num_classes=self.num_classes, freeze_backbone=True)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Filter the checkpoint state_dict
        model_state_dict = model.state_dict()
        
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"⚠️ Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)
        
        if skipped_keys:
            print(f"\n⚠️ Skipped {len(skipped_keys)} keys from checkpoint")
            for key in skipped_keys[:10]:
                print(f"   Skipped: {key}")
            if len(skipped_keys) > 10:
                print(f"   ... and {len(skipped_keys) - 10} more")
        
        # Load the filtered state_dict with strict=False
        model.load_state_dict(filtered_state_dict, strict=False)
        model = model.to(self.device)
        
        print(f"✅ Model loaded from {model_path}")
        return model
    
    def get_attention(self, image_tensor):
        """Get attention maps for a given image tensor."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(image_tensor)
        
        return attention_maps
    
    def get_attention_resized(self, image_tensor, target_size=(224, 224)):
        """Get attention maps resized to target size."""
        attn = self.get_attention(image_tensor)
        attn_resized = torch.nn.functional.interpolate(
            attn, size=target_size, mode='bilinear', align_corners=False
        )
        return attn_resized


def create_effnetb3_attention_model(num_classes=3):
    """Create EfficientNet-B3 + Attention model."""
    model = EfficientNetB3AttentionModel(num_classes=num_classes, freeze_backbone=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return model


def load_effnetb3_attention_model(model_path, num_classes=3, device='cpu'):
    """Load a trained EfficientNet-B3 + Attention model."""
    print(f"Loading model from {model_path}")
    
    model = create_effnetb3_attention_model(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except:
        checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    # Filter the checkpoint state_dict
    model_state_dict = model.state_dict()
    
    filtered_state_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
            else:
                print(f"⚠️ Shape mismatch for {key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
                skipped_keys.append(key)
        else:
            skipped_keys.append(key)
    
    if skipped_keys:
        print(f"\n⚠️ Skipped {len(skipped_keys)} keys from checkpoint")
        for key in skipped_keys[:10]:
            print(f"   Skipped: {key}")
        if len(skipped_keys) > 10:
            print(f"   ... and {len(skipped_keys) - 10} more")
    
    # Load the filtered state_dict with strict=False
    model.load_state_dict(filtered_state_dict, strict=False)
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
    parser = argparse.ArgumentParser(description="Extract attention maps from EfficientNet-B3+Attention model")
    
    # Required arguments
    parser.add_argument("-m", "--model_path", type=str, required=True, 
                        help="Path to the trained model (.pt file)")
    
    # Optional arguments
    parser.add_argument("-f", "--fold", type=int, default=1,
                        help="Fold number to use (default: 1)")
    parser.add_argument("-n", "--num_samples", type=int, default=50,
                        help="Number of samples to visualize (default: 50)")
    parser.add_argument("-s", "--start_idx", type=int, default=0,
                        help="Starting index for image selection (default: 0)")
    
    # Selection mode
    parser.add_argument("--random", action='store_true',
                        help="Select images randomly (default: sequential)")
    parser.add_argument("--by_class", action='store_true',
                        help="Select equal number of images per class")
    
    # Output options
    parser.add_argument("-o", "--output_dir", type=str, default="effnetb3_attention",
                        help="Output directory name (default: effnetb3_attention)")
    parser.add_argument("--no_save", action='store_true',
                        help="Don't save images, just display")
    
    # Visualization style
    parser.add_argument("--style", type=str, default="combined", 
                        choices=['single', 'opencv', 'combined'],
                        help="Visualization style: single, opencv, or combined (default: combined)")
    
    # Tiling options
    parser.add_argument("--tile_size", type=int, default=224,
                        help="Size of each tile in pixels (default: 224)")
    parser.add_argument("--grid_size", type=int, default=2,
                        help="Grid size for tiling (e.g., 2 for 2x2 grid, default: 2)")
    parser.add_argument("--alpha_overlay", type=float, default=0.3,
                        help="Alpha value for overlay (default: 0.3)")
    
    return parser.parse_args()

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

def select_images(test_images, test_labels, num_samples, start_idx=0, random_select=False, by_class=False):
    """
    Select images based on the specified mode.
    """
    n_total = len(test_images)
    
    if by_class:
        unique_classes = np.unique(test_labels)
        n_classes = len(unique_classes)
        samples_per_class = num_samples // n_classes
        
        selected_indices = []
        for cls in unique_classes:
            cls_indices = np.where(test_labels == cls)[0]
            if random_select:
                if len(cls_indices) > samples_per_class:
                    cls_selected = np.random.choice(cls_indices, samples_per_class, replace=False)
                else:
                    cls_selected = cls_indices
            else:
                if len(cls_indices) > samples_per_class:
                    offset = (start_idx * (cls + 1)) % len(cls_indices)
                    end_idx = min(offset + samples_per_class, len(cls_indices))
                    if end_idx == len(cls_indices):
                        cls_selected = np.concatenate([
                            cls_indices[offset:],
                            cls_indices[:samples_per_class - (len(cls_indices) - offset)]
                        ])
                    else:
                        cls_selected = cls_indices[offset:end_idx]
                else:
                    cls_selected = cls_indices
            selected_indices.extend(cls_selected)
        
        selected_indices = np.array(selected_indices)
        
    else:
        if random_select:
            if n_total > num_samples:
                selected_indices = np.random.choice(n_total, num_samples, replace=False)
            else:
                selected_indices = np.arange(n_total)
        else:
            if n_total > num_samples:
                end_idx = min(start_idx + num_samples, n_total)
                if end_idx < n_total:
                    selected_indices = np.arange(start_idx, end_idx)
                else:
                    remaining = num_samples - (n_total - start_idx)
                    selected_indices = np.concatenate([
                        np.arange(start_idx, n_total),
                        np.arange(remaining)
                    ])
            else:
                selected_indices = np.arange(n_total)
    
    selected_images = test_images[selected_indices]
    selected_labels = test_labels[selected_indices]
    
    print(f"\n📊 Selection Summary:")
    print(f"  Total available: {n_total}")
    print(f"  Selected: {len(selected_indices)}")
    print(f"  Mode: {'Random' if random_select else 'Sequential'}")
    print(f"  Balanced by class: {by_class}")
    
    if by_class:
        unique, counts = np.unique(selected_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls} ({class_names[cls]}): {count} images")
    
    return selected_images, selected_labels, selected_indices


# ==================== ATTENTION SAVE FUNCTIONS ====================
def save_attention_image(img, attention, target, prediction, i, arch, save_dir=None, display=False):
    """Save attention map visualization - Original style."""
    if save_dir is None:
        save_dir = f"../experiments/{arch}/attention_maps/"
    os.makedirs(save_dir, exist_ok=True)
    
    if attention.ndim == 3:
        attention = attention.squeeze(0)
    
    attention_resized = np.array(Image.fromarray(attention).resize((img.shape[1], img.shape[0]), 
                                                                    Image.Resampling.BILINEAR))
    
    attention_resized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    
    title = f"True: {target} | Pred: {prediction}"
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(title, fontsize=16, y=0.85)
    fig.set_size_inches(18.5, 10.5)
    
    axs[0].imshow(img.astype(np.uint8))
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")
    
    im = axs[1].imshow(attention_resized, cmap='viridis')
    axs[1].set_title("Attention Map (7x7 -> 224x224)", fontsize=14)
    axs[1].axis("off")
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    axs[2].imshow(img.astype(np.uint8))
    axs[2].imshow(attention_resized, alpha=alpha, cmap='viridis', vmin=0, vmax=1)
    axs[2].set_title("Overlay", fontsize=14)
    axs[2].axis("off")
    
    if not display:
        fig.savefig(f"{save_dir}/image_{i:04d}.png", bbox_inches='tight', dpi=150)
        plt.clf()
    plt.close()


def save_attention_image_opencv_style(img, attention, target, prediction, i, arch, 
                                       save_dir=None, display=False, tile_size=224, 
                                       grid_size=2, alpha_overlay=0.3):
    """Save attention map visualization using OpenCV-style tiling and overlay."""
    if save_dir is None:
        save_dir = f"../experiments/{arch}/attention_maps/"
    os.makedirs(save_dir, exist_ok=True)
    
    if attention.ndim == 3:
        attention = attention.squeeze(0)
    
    h, w = img.shape[:2]
    
    attention_resized = np.array(Image.fromarray(attention).resize((tile_size, tile_size), 
                                                                    Image.Resampling.BILINEAR))
    
    attn_min = attention_resized.min()
    attn_max = attention_resized.max()
    if attn_max > attn_min:
        attention_resized = (attention_resized - attn_min) / (attn_max - attn_min + 1e-8)
    else:
        attention_resized = np.zeros_like(attention_resized)
    
    # Tiled heatmap
    big_heatmap = np.zeros((grid_size * tile_size, grid_size * tile_size))
    big_heatmap_only = np.zeros((grid_size * tile_size, grid_size * tile_size, 3))
    
    for i_tile in range(grid_size):
        for j_tile in range(grid_size):
            heatmap_tile = attention_resized.copy()
            
            y_start = i_tile * tile_size
            y_end = (i_tile + 1) * tile_size
            x_start = j_tile * tile_size
            x_end = (j_tile + 1) * tile_size
            
            big_heatmap[y_start:y_end, x_start:x_end] = heatmap_tile
            
            stacked_img = np.stack((heatmap_tile,), axis=-1)
            im_color = cv2.applyColorMap((stacked_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            big_heatmap_only[y_start:y_end, x_start:x_end] = im_color
    
    big_img = np.array(Image.fromarray(img).resize((grid_size * tile_size, grid_size * tile_size),
                                                    Image.Resampling.BILINEAR))
    
    superimposed_img = cv2.addWeighted(big_heatmap_only.astype(np.uint8), alpha_overlay, 
                                       big_img.astype(np.uint8), 1 - alpha_overlay, 0)
    
    # Single tile overlay
    attn_resized_full = np.array(Image.fromarray(attention).resize((w, h), 
                                                                    Image.Resampling.BILINEAR))
    attn_min = attn_resized_full.min()
    attn_max = attn_resized_full.max()
    if attn_max > attn_min:
        attn_resized_full = (attn_resized_full - attn_min) / (attn_max - attn_min + 1e-8)
    else:
        attn_resized_full = np.zeros_like(attn_resized_full)
    
    stacked_img_full = np.stack((attn_resized_full,), axis=-1)
    im_color_full = cv2.applyColorMap((stacked_img_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_img_full = cv2.addWeighted(im_color_full.astype(np.uint8), alpha_overlay, 
                                            img.astype(np.uint8), 1 - alpha_overlay, 0)
    
    title = f"True: {target} | Pred: {prediction}"
    
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    axs[0, 0].imshow(img.astype(np.uint8))
    axs[0, 0].set_title("Original Image", fontsize=14)
    axs[0, 0].axis("off")
    
    im1 = axs[0, 1].imshow(attn_resized_full, cmap='viridis')
    axs[0, 1].set_title("Attention Map (Single)", fontsize=14)
    axs[0, 1].axis("off")
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    
    axs[0, 2].imshow(superimposed_img_full.astype(np.uint8))
    axs[0, 2].set_title("Overlay (Single Tile)", fontsize=14)
    axs[0, 2].axis("off")
    
    im2 = axs[1, 0].imshow(big_heatmap, cmap='viridis')
    axs[1, 0].set_title(f"Tiled Heatmap ({grid_size}x{grid_size} tiles)", fontsize=14)
    axs[1, 0].axis("off")
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    
    axs[1, 1].imshow(superimposed_img.astype(np.uint8))
    axs[1, 1].set_title("Tiled Overlay (OpenCV Style)", fontsize=14)
    axs[1, 1].axis("off")
    
    axs[1, 2].axis("off")
    info_text = (
        f"Model: EfficientNet-B3\n"
        f"Tile Size: {tile_size}x{tile_size}\n"
        f"Grid: {grid_size}x{grid_size}\n"
        f"Total: {grid_size**2} tiles\n"
        f"Alpha: {alpha_overlay}\n"
        f"Colormap: JET"
    )
    axs[1, 2].text(0.1, 0.5, info_text, fontsize=14, verticalalignment='center',
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    if not display:
        fig.savefig(f"{save_dir}/image_{i:04d}_opencv_style.png", bbox_inches='tight', dpi=150)
        print(f"Saved: {save_dir}/image_{i:04d}_opencv_style.png")
    else:
        plt.show()
    plt.close()


def save_attention_image_combined(img, attention, target, prediction, i, arch, 
                                   save_dir=None, display=False, tile_size=224, 
                                   grid_size=2, alpha_overlay=0.3):
    """Combined visualization showing both original and OpenCV-style attention maps."""
    if save_dir is None:
        save_dir = f"../experiments/{arch}/attention_maps/"
    os.makedirs(save_dir, exist_ok=True)
    
    if attention.ndim == 3:
        attention = attention.squeeze(0)
    
    h, w = img.shape[:2]
    
    attention_resized = np.array(Image.fromarray(attention).resize((tile_size, tile_size), 
                                                                    Image.Resampling.BILINEAR))
    
    attn_min = attention_resized.min()
    attn_max = attention_resized.max()
    if attn_max > attn_min:
        attention_resized = (attention_resized - attn_min) / (attn_max - attn_min + 1e-8)
    else:
        attention_resized = np.zeros_like(attention_resized)
    
    # Create tiled heatmap
    big_heatmap = np.zeros((grid_size * tile_size, grid_size * tile_size))
    big_heatmap_only = np.zeros((grid_size * tile_size, grid_size * tile_size, 3))
    big_img = np.array(Image.fromarray(img).resize((grid_size * tile_size, grid_size * tile_size),
                                                    Image.Resampling.BILINEAR))
    
    for i_tile in range(grid_size):
        for j_tile in range(grid_size):
            heatmap_tile = attention_resized.copy()
            
            y_start = i_tile * tile_size
            y_end = (i_tile + 1) * tile_size
            x_start = j_tile * tile_size
            x_end = (j_tile + 1) * tile_size
            
            big_heatmap[y_start:y_end, x_start:x_end] = heatmap_tile
            
            stacked_img = np.stack((heatmap_tile,), axis=-1)
            im_color = cv2.applyColorMap((stacked_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
            big_heatmap_only[y_start:y_end, x_start:x_end] = im_color
    
    superimposed_img = cv2.addWeighted(big_heatmap_only.astype(np.uint8), alpha_overlay, 
                                       big_img.astype(np.uint8), 1 - alpha_overlay, 0)
    
    # Single tile overlay
    attn_resized_full = np.array(Image.fromarray(attention).resize((w, h), 
                                                                    Image.Resampling.BILINEAR))
    attn_min = attn_resized_full.min()
    attn_max = attn_resized_full.max()
    if attn_max > attn_min:
        attn_resized_full = (attn_resized_full - attn_min) / (attn_max - attn_min + 1e-8)
    else:
        attn_resized_full = np.zeros_like(attn_resized_full)
    
    stacked_img_full = np.stack((attn_resized_full,), axis=-1)
    im_color_full = cv2.applyColorMap((stacked_img_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_img_full = cv2.addWeighted(im_color_full.astype(np.uint8), alpha_overlay, 
                                            img.astype(np.uint8), 1 - alpha_overlay, 0)
    
    title = f"True: {target} | Pred: {prediction}"
    
    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(16, 14))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    axs[0, 0].imshow(big_img.astype(np.uint8))
    axs[0, 0].set_title(f"Original Image ({grid_size}x{grid_size} tiles)", fontsize=14)
    axs[0, 0].axis("off")
    
    im_attn = axs[0, 1].imshow(attention_resized, cmap='viridis')
    axs[0, 1].set_title(f"Attention Map ({tile_size}x{tile_size})", fontsize=14)
    axs[0, 1].axis("off")
    plt.colorbar(im_attn, ax=axs[0, 1], fraction=0.046, pad=0.04)
    
    axs[1, 0].imshow(superimposed_img.astype(np.uint8))
    axs[1, 0].set_title("Tiled Overlay (OpenCV Style)", fontsize=14)
    axs[1, 0].axis("off")
    
    axs[1, 1].imshow(superimposed_img_full.astype(np.uint8))
    axs[1, 1].set_title("Single Overlay (Original Style)", fontsize=14)
    axs[1, 1].axis("off")
    
    info_text = f"Model: EfficientNet-B3 | Tile: {tile_size} | Grid: {grid_size}x{grid_size} | Alpha: {alpha_overlay}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
    if not display:
        fig.savefig(f"{save_dir}/image_{i:04d}_combined.png", bbox_inches='tight', dpi=150)
        print(f"Saved: {save_dir}/image_{i:04d}_combined.png")
    else:
        plt.show()
    plt.close()


# ==================== COMPUTE ATTENTION MAPS FUNCTION ====================
def compute_attention_maps(device, model_path, fold=1, num_samples=50, start_idx=0, 
                           random_select=False, by_class=False, output_dir="effnetb3_attention", 
                           no_save=False, style='combined', tile_size=224, 
                           grid_size=2, alpha_overlay=0.3):
    """
    Compute and visualize attention maps from the EfficientNet-B3 + Attention model.
    """
    # Load data
    images, labels = load_data()
    
    # Load model
    model = load_effnetb3_attention_model(model_path, num_classes=nclasses, device=device)
    
    if model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    model.eval()
    
    # Use StratifiedKFold to get test indices
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    folds = list(skf.split(images, labels))
    
    _, test_index = folds[fold - 1]
    
    test_images = images[test_index]
    test_labels = labels[test_index]
    
    # Select images based on mode
    selected_images, selected_labels, selected_indices = select_images(
        test_images, test_labels, num_samples, start_idx, random_select, by_class
    )
    
    print(f"\nGenerating attention maps for {len(selected_images)} images...")
    print(f"Attention map size: 7x7 (EfficientNet-B3 feature map)")
    print(f"Visualization style: {style}")
    print(f"Tile size: {tile_size}, Grid: {grid_size}x{grid_size}")
    
    # Create attention extractor
    extractor = AttentionExtractor(model_path, num_classes=nclasses, device=device)
    
    correct = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(zip(selected_images, selected_labels), 1):
            # Prepare image
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            img_norm = normalize(img_tensor / 255, mean, std)
            img_input = img_norm.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(img_input)
            _, pred = torch.max(output, 1)
            
            if pred.item() == label:
                correct += 1
            
            # Get attention map
            attn_map = extractor.get_attention(img_input)  # [B, 1, 7, 7]
            attn = attn_map.squeeze().cpu().numpy()  # [7, 7]
            
            # Apply filters for smoothing
            attn = median_filter(attn, footprint=morphology.disk(patch_size // 2))
            attn = gaussian_filter(attn, sigma=patch_size // 4)
            
            # Get labels
            true_label = mapping_reverse[label]
            pred_label = mapping_reverse[int(pred)]
            
            # Save based on style
            if style == 'single':
                save_func = save_attention_image
                if no_save:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, display=True)
                else:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, display=False)
            elif style == 'opencv':
                save_func = save_attention_image_opencv_style
                if no_save:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, 
                             display=True, tile_size=tile_size, grid_size=grid_size, 
                             alpha_overlay=alpha_overlay)
                else:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, 
                             display=False, tile_size=tile_size, grid_size=grid_size, 
                             alpha_overlay=alpha_overlay)
            else:  # combined (default)
                save_func = save_attention_image_combined
                if no_save:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, 
                             display=True, tile_size=tile_size, grid_size=grid_size, 
                             alpha_overlay=alpha_overlay)
                else:
                    save_func(img, attn, true_label, pred_label, idx, output_dir, 
                             display=False, tile_size=tile_size, grid_size=grid_size, 
                             alpha_overlay=alpha_overlay)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(selected_images)} images...")
    
    accuracy = correct / len(selected_images) * 100
    print(f"\n✅ Attention maps processed!")
    if not no_save:
        print(f"   Saved to ../experiments/{output_dir}/attention_maps/")
    print(f"   Model accuracy on selected samples: {accuracy:.2f}%")


# ==================== MAIN FUNCTION ====================
def main():
    device, devname = identify_device(seed)
    args = parse_arguments()
    
    print(f"================================================================")
    print(f"Using {device} - {devname}")
    print(f"Model: EfficientNet-B3 + Attention")
    print(f"Architecture: 1536 -> 32 -> 8 -> 1 -> 512 -> {nclasses}")
    print(f"Model Path: {args.model_path}")
    print(f"Fold: {args.fold}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Start index: {args.start_idx}")
    print(f"Random selection: {args.random}")
    print(f"Balanced by class: {args.by_class}")
    print(f"Output directory: {args.output_dir}")
    print(f"Visualization style: {args.style}")
    print(f"Tile size: {args.tile_size}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Alpha overlay: {args.alpha_overlay}")
    print(f"================================================================")
    print()
    
    compute_attention_maps(
        device=device,
        model_path=args.model_path,
        fold=args.fold,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        random_select=args.random,
        by_class=args.by_class,
        output_dir=args.output_dir,
        no_save=args.no_save,
        style=args.style,
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        alpha_overlay=args.alpha_overlay
    )
    exit(0)

if __name__ == "__main__":
    main()