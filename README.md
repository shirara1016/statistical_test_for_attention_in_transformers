# Statistical Test for Attention in Transformers for Images and Time Series
This package is the implementation of the paper "Statistical Test for Attention in Transformers for Images and Time Series" for experiments.

## Installation & Requirements
This package has the following dependencies:
- Python (version 3.10 or higher, we use 3.12.5)
    - torch (version 2.5.0 or higher, we use 2.5.0)
    - sicore (version 2.2.0 or higher, we use 2.2.0)
    - polars (version 1.7.1 or higher, we use 1.11.0)
    - tqdm (version 4.66.5 or higher, we use 4.66.5)

Please install these dependencies by pip.
```bash
pip install torch
pip install sicore # note that other dependencies will be installed automatically
pip install polars
pip install tqdm
```

## Reproducibility
To reproduce the results, please see the following instructions after installation step.
The results will be saved in `./result` folder as csv files.
The plots will be saved in `./figures` folder as pdf files, which we have already got in advance.

For reproducing the Figures 4 and 5 (type I error rate).
```bash
bash exp_fpr_images.sh
bash exp_fpr_series.sh
```

For reproducing the Figure 6 (power).
```bash
bash exp_tpr.sh
```

For reproducing the Figures 7 and 8 (computation time).
```bash
bash exp_time_images.sh
bash exp_time_series.sh
```

For visualization of the reproduced results.
```bash
bash summary.sh
```
