!pip install webdataset huggingface_hub -q

# =============================================================================
# Phase 1: Extract patch-level ResNet-18 embeddings for all TCGA-UT patients
# Phase 2: Save patch embeddings to disk (.npz per patient)
# Phase 3: Derive risk labels from clinical data (self-contained, no Databricks needed)
# Phase 4: Train ABMIL (Attention-Based MIL) to learn patient-level representations
# Phase 5: Export attention-pooled 512-dim vectors as CSV for Databricks upload
# =============================================================================

import torch, torch.nn as nn, torchvision.models as models, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd, numpy as np, io, os, tarfile
from huggingface_hub import hf_hub_download
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# Phase 1: ResNet-18 feature extraction
# =============================================================================

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("ResNet-18 loaded")

repo_id = "dakomura/tcga-ut"
shards = []
for i in range(39):
    shards.append(f"data/dataset_internal_train_part{str(i).zfill(3)}.tar")
for i in range(6):
    shards.append(f"data/dataset_internal_valid_part{str(i).zfill(3)}.tar")
for i in range(6):
    shards.append(f"data/dataset_internal_test_part{str(i).zfill(3)}.tar")

os.makedirs("/content/shards", exist_ok=True)
for s in shards:
    print(f"Downloading {s}...")
    hf_hub_download(repo_id=repo_id, filename=s, repo_type="dataset", local_dir="/content/shards")
print(f"Downloads complete. Total shards: {len(shards)}")

patient_embeddings = defaultdict(list)
processed = errors = 0
for s in shards:
    print(f"\nProcessing {s}...")
    with tarfile.open(f"/content/shards/{s}", "r") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".jpg"): continue
            bp = None
            for p in m.name.split("/"):
                if p.startswith("TCGA-"): bp = p; break
            if not bp: continue
            try:
                img = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
                with torch.no_grad():
                    emb = resnet(transform(img).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                patient_embeddings[bp[:12]].append(emb)
                processed += 1
                if processed % 10000 == 0: print(f"  {processed} tiles, {len(patient_embeddings)} patients")
            except Exception as e:
                errors += 1
                if errors <= 5: print(f"  Error: {e}")
print(f"\nDone: {processed} tiles, {errors} errors, {len(patient_embeddings)} patients")

# =============================================================================
# Phase 2: Save patch-level embeddings to disk
# =============================================================================

os.makedirs("/content/patch_embeddings", exist_ok=True)
for barcode, emb_list in patient_embeddings.items():
    np.save(f"/content/patch_embeddings/{barcode}.npy", np.array(emb_list))
print(f"Saved patch embeddings for {len(patient_embeddings)} patients to /content/patch_embeddings/")

# =============================================================================
# Phase 3: Derive risk labels (self-contained, no Databricks needed)
# =============================================================================

# Download clinical data CSV — pre-staged or from the same source
# Expects TCGA-CDR with columns: bcr_patient_barcode, type, OS, OS.time
!pip install openpyxl -q

clinical_path = "/content/TCGA-CDR-SupplementalTableS1.xlsx"
if not os.path.exists(clinical_path):
    print("Upload TCGA-CDR-SupplementalTableS1.xlsx to /content/ before running this cell")
    raise FileNotFoundError(clinical_path)

cdr = pd.read_excel(clinical_path, sheet_name="TCGA-CDR", engine="openpyxl")
cdr = cdr.rename(columns={"bcr_patient_barcode": "patient_barcode"})
cdr["OS"] = pd.to_numeric(cdr["OS"], errors="coerce")
cdr["OS.time"] = pd.to_numeric(cdr["OS.time"], errors="coerce")

# Median OS per cancer type among deceased patients only
deceased = cdr[cdr["OS"] == 1].groupby("type")["OS.time"].median().rename("median_os")
cdr = cdr.merge(deceased, left_on="type", right_index=True, how="left")

# Derive risk labels
def label_risk(row):
    if row["OS"] == 1 and row["OS.time"] < row["median_os"]:
        return "high_risk"
    elif row["OS.time"] >= row["median_os"]:
        return "low_risk"
    else:
        return "censored"

cdr["risk_label"] = cdr.apply(label_risk, axis=1)

# Only keep patients we have embeddings for, and exclude censored
labeled = cdr[cdr["patient_barcode"].isin(patient_embeddings.keys())]
labeled = labeled[labeled["risk_label"].isin(["high_risk", "low_risk"])].reset_index(drop=True)
label_map = {"high_risk": 1, "low_risk": 0}
labeled["y"] = labeled["risk_label"].map(label_map)

print(f"Labeled patients with embeddings: {len(labeled)}")
print(labeled["risk_label"].value_counts())

# =============================================================================
# Phase 4: Train ABMIL
# =============================================================================

MAX_PATCHES = 200  # sample this many patches per patient (pad if fewer)
EMB_DIM = 512

class PatientBagDataset(Dataset):
    def __init__(self, barcodes, labels, emb_dir, max_patches=MAX_PATCHES):
        self.barcodes = barcodes
        self.labels = labels
        self.emb_dir = emb_dir
        self.max_patches = max_patches

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        barcode = self.barcodes[idx]
        patches = np.load(f"{self.emb_dir}/{barcode}.npy")  # (N, 512)

        # Sample or pad to fixed size
        n = patches.shape[0]
        if n > self.max_patches:
            indices = np.random.choice(n, self.max_patches, replace=False)
            patches = patches[indices]
        elif n < self.max_patches:
            pad = np.zeros((self.max_patches - n, EMB_DIM), dtype=np.float32)
            patches = np.concatenate([patches, pad], axis=0)

        mask = np.zeros(self.max_patches, dtype=np.float32)
        mask[:min(n, self.max_patches)] = 1.0

        return (
            torch.tensor(patches, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class GatedAttentionMIL(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x, mask):
        # x: (batch, max_patches, 512), mask: (batch, max_patches)
        V = self.attention_V(x)  # (batch, max_patches, hidden)
        U = self.attention_U(x)  # (batch, max_patches, hidden)
        scores = self.attention_w(V * U).squeeze(-1)  # (batch, max_patches)

        # Mask out padded patches with large negative value
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)  # (batch, max_patches)

        # Attention-weighted pooling
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, 512)

        logits = self.classifier(pooled)
        return logits, pooled, weights


# Train/test split — use 80/20 stratified
from sklearn.model_selection import train_test_split

train_labeled, test_labeled = train_test_split(
    labeled, test_size=0.2, random_state=42, stratify=labeled["y"]
)

train_ds = PatientBagDataset(
    train_labeled["patient_barcode"].values,
    train_labeled["y"].values,
    "/content/patch_embeddings",
)
test_ds = PatientBagDataset(
    test_labeled["patient_barcode"].values,
    test_labeled["y"].values,
    "/content/patch_embeddings",
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# Class weights for imbalanced labels
class_counts = np.bincount(train_labeled["y"].values)
class_weights = torch.tensor(
    [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * 2

model = GatedAttentionMIL().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

print(f"Training ABMIL: {len(train_ds)} train, {len(test_ds)} test patients")

for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for patches, mask, labels in train_loader:
        patches, mask, labels = patches.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(patches, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    if (epoch + 1) % 5 == 0:
        train_acc = correct / total
        # Eval
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for patches, mask, labels in test_loader:
                patches, mask, labels = patches.to(device), mask.to(device), labels.to(device)
                logits, _, _ = model(patches, mask)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += len(labels)
        test_acc = test_correct / test_total
        print(f"  Epoch {epoch+1:>2}/30  loss={total_loss/total:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

print("ABMIL training complete.")

# =============================================================================
# Phase 5: Extract attention-pooled vectors for ALL patients (including censored)
# =============================================================================

model.eval()
all_barcodes = list(patient_embeddings.keys())

all_ds = PatientBagDataset(
    np.array(all_barcodes),
    np.zeros(len(all_barcodes), dtype=np.int64),  # dummy labels
    "/content/patch_embeddings",
)
all_loader = DataLoader(all_ds, batch_size=64, shuffle=False)

pooled_vectors = []
with torch.no_grad():
    for patches, mask, _ in all_loader:
        patches, mask = patches.to(device), mask.to(device)
        _, pooled, _ = model(patches, mask)
        pooled_vectors.append(pooled.cpu().numpy())

pooled_matrix = np.concatenate(pooled_vectors, axis=0)  # (num_patients, 512)

rows = []
for i, barcode in enumerate(all_barcodes):
    row = {"patient_barcode": barcode}
    for j in range(EMB_DIM):
        row[f"img_emb_{j}"] = float(pooled_matrix[i, j])
    rows.append(row)

df = pd.DataFrame(rows)
print(f"Shape: {df.shape}  (512 attention-pooled features per patient)")
df.to_csv("/content/patient_image_embeddings_full.csv", index=False)
print(f"Saved. Size: {os.path.getsize('/content/patient_image_embeddings_full.csv')/1024/1024:.1f} MB")

from google.colab import files
files.download("/content/patient_image_embeddings_full.csv")
