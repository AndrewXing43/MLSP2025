import os
import numpy as np
from PIL import Image
from pathlib import Path
 
def generate_sparse_samples(image_path, sampling_rate):
     img = Image.open(image_path).convert("L")
     pl_array = np.array(img)
     h, w = pl_array.shape
     num_samples = int(np.ceil(sampling_rate * h * w))
 
     xs, ys = np.unravel_index(np.random.choice(h * w, num_samples, replace=False), (h, w))
     samples = np.stack([xs, ys, pl_array[xs, ys]], axis=1)
     return samples
 
def process_all_images(outputs_dir, output_dir_05, output_dir_002):
     outputs_dir = Path(outputs_dir)
     output_05_dir = Path(output_dir_05)
     output_002_dir = Path(output_dir_002)
 
     output_05_dir.mkdir(parents=True, exist_ok=True)
     output_002_dir.mkdir(parents=True, exist_ok=True)
 
     for img_file in outputs_dir.glob("*.png"):
         samples_05 = generate_sparse_samples(img_file, sampling_rate=0.005)
         samples_002 = generate_sparse_samples(img_file, sampling_rate=0.0002)
 
         np.save(output_05_dir / (img_file.stem + ".npy"), samples_002)
         #np.save(output_002_dir / (img_file.stem + ".npy"), samples_002)
 
 # Example usage:
process_all_images("/home/andrew43/competition/outputs","/home/andrew43/competition/sparse_samples_0.5", "/home/andrew43/competition/sparse_samples_0.02")