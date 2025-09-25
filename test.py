import gdown
import os

# File IDs
file_ids = [
    "18fG_uxXnRYwPlf3WkvYSeSHo6f4dW7dE",  # Tumhari pehli file ID
    "1D4Em7-OsLDT6M39Nj_6ALfiJBDa6FxVR"   # Tumhari doosri file ID
]

# Destination folder
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Download files
for file_id in file_ids:
    output_path = os.path.join(output_dir, f"{file_id}.h5ad")  # Ya .csv, jo bhi file type ho
    if not os.path.exists(output_path):
        print(f"Downloading file {file_id}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t", output_path, quiet=False)
    else:
        print(f"File {file_id} already exists. Skipping download.")
print("All files downloaded.")