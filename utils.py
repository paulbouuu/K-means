import os
import re
import glob
import numpy as np
from PIL import Image


def get_iteration_index(filename):
    """Extract the numeric iteration from a filename like 'k_means_7.png'."""
    match = re.search(r'k_means_(\d+)\.png$', filename)
    if not match:
        raise ValueError(f"Filename does not match pattern: {filename}")
    return int(match.group(1))

def gif(duration = 500):
    img_dir = 'images'
    pattern = os.path.join(img_dir, 'k_means_*.png')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching {pattern}")
        return

    files_sorted = sorted(files, key=get_iteration_index)
    
    print(f"Found {len(files_sorted)} frames.")

    frames = []
    for f in files_sorted:
        try:
            img = Image.open(f)
            frames.append(img.convert('P', palette=Image.ADAPTIVE))
        except Exception as e:
            print(f"Warning: could not open {f}: {e}")

    if not frames:
        print("No valid images to combine. Exiting.")
        return

    # save as animated GIF
    output_path = 'k_means_anim.gif'
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,   # milliseconds per frame; adjust as needed
            loop=0,         # 0 = infinite loop
            optimize=True
        )
        print(f"Saved GIF to {output_path}")
    except Exception as e:
        print(f"Error writing GIF: {e}")

def generate_2D_clusters(k = 3, points_per_cluster = 200, domain = [-5, 5]):
    """
    Generate a mixture of 2D Gaussians

    Inputs:
        - k : number of clusters
        - points_per_cluster : number of points per cluster
        - domain : the domain of the clusters

    Outputs:
        - X : the generated points
    """
    means = np.random.rand(k, 2) * (domain[1] - domain[0]) + domain[0]
    covariances = []

    for _ in range(k):
        # Random eigenvalues for width and height
        eigvals = np.random.uniform(0.2, 1.0, size=2)

        # Small random rotation angle
        angle = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])

        cov = R @ np.diag(eigvals) @ R.T / 2
        covariances.append(cov)

    X = [np.random.multivariate_normal(means[i], covariances[i], points_per_cluster) for i in range(k)]
    return np.vstack(X)