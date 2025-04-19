# K‑means Clustering in 2D

This project implements the classic K‑means algorithm from scratch in Python for 2D data. It includes step‑by‑step visualizations of how the cluster centroids move over iterations, and saves each frame as an image for optional animation.

The core class `K_means` is defined in [k_means.py](k_means.py). An example of how to generate data, run the algorithm, and visualize the results is provided in the Jupyter notebook [k_means_demo.ipynb](k_means_demo.ipynb).

---

## 🚀 Features

- Stepwise visualization: saves a PNG at each iteration in `images/`  
- Optional creation of a GIF animation
- Minimal dependencies

---

## 📁 Repository Structure

```
.
├── k_means.py            # EM algorithm for MoG
├── utils.py              # GIF generation & data generation
├── images/               # Saved plots for each iteration (will be created during the first run)
├── resources/            # GIF examples
├── k_means_anim.gif      # Output animation (will be created during the first run)
├── k_means_demo.ipynb    # Example usage in Jupyter
└── README.md
```

---

## 📦 Requirements

- Numpy
- Matplotlib
- Pillow  (if you set `create_gif=True`)

---

## 🧪 How to Run

You can test the algorithm and generate visualizations with the included notebook `k_means_demo.ipynb`. This repo has been designed to be simple to understand and modify.

## Documentation:

**_class_ `K_means`:**
- **Parameters:**
    - `k` (`int`): number of clusters, default is `3`
    - `domain` (`list`): domain of the data, default is `[-5, 5]`

**Other parameters:**
- `iterations` : number of iterations of the K-means algorithm
- `plotting` : whether to display the plot interactively
- `create_gif` : whether to create a GIF animation of the results (needs pillow package)

## 📈 Output

The class `K_means` saves visualizations of each iteration in the `images/` folder. Once training completes, it generates an animated GIF (`k_means_anim.gif`) illustrating how the centroids fit the data over time.

![K-means Animation](https://github.com/paulbouuu/K-means/raw/main/resources/optimal_k_means.gif)

### License
This project is free to use and modify under the MIT License. See the [LICENSE](LICENSE) file for details.
