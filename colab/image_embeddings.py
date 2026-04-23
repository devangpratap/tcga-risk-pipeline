!pip install webdataset huggingface_hub -q

import torch, torch.nn as nn, torchvision.models as models, torchvision.transforms as transforms
from PIL import Image
import pandas as pd, numpy as np, io, os, tarfile
from huggingface_hub import hf_hub_download
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
shards = [f"data/dataset_internal_train_part{str(i).zfill(3)}.tar" for i in range(5)]
os.makedirs("/content/shards", exist_ok=True)
for s in shards:
    print(f"Downloading {s}...")
    hf_hub_download(repo_id=repo_id, filename=s, repo_type="dataset", local_dir="/content/shards")
print("Downloads complete.")

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
                if processed % 5000 == 0: print(f"  {processed} tiles, {len(patient_embeddings)} patients")
            except: errors += 1
print(f"\nDone: {processed} tiles, {errors} errors, {len(patient_embeddings)} patients")

rows = []
for b, el in patient_embeddings.items():
    row = {"patient_barcode": b}
    for i, v in enumerate(np.mean(el, axis=0)): row[f"img_emb_{i}"] = float(v)
    rows.append(row)
df = pd.DataFrame(rows)
print(f"Shape: {df.shape}")
df.to_csv("/content/patient_image_embeddings.csv", index=False)
print(f"Saved. Size: {os.path.getsize('/content/patient_image_embeddings.csv')/1024/1024:.1f} MB")

from google.colab import files
files.download("/content/patient_image_embeddings.csv")
