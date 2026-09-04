# Joint Hyperspectral Image Deconvolution and Unmixing via Plug-and-Play Priors

Plug-and-Play Joint Deblurring and Unmixing for Hyperspectral Imaging.

This repository contains the Python implementation for the PnP-JDU method described in the published paper:

- DOI: https://doi.org/10.3390/rs18132066
- Article: "Plug-and-Play Joint Deblurring and Unmixing for Hyperspectral Imaging"

## Overview

The code evaluates a joint deblurring and unmixing framework for hyperspectral image reconstruction using:

- Indian Pines dataset
- Six blur kernel configurations
- PnP-JDU, FCLS, UCLS, NNLS, and PnP baselines
- Gaussian-noise experiments and output metrics

## Repository contents

- `PnPJDU.py` — main experiment script
- `dataset/` — Indian Pines dataset files
- `Kernels/` — blur kernels used in the paper
- `remotesensing-18-02066-v2.pdf` — published paper PDF

## Requirements

Install the project dependencies before running the experiment, including:

- Python 3.9+
- JAX
- SciPy
- scico
- scikit-image
- spectral
- pysptools
- pandas
- NumPy
- matplotlib

## Run

From the project root:

```bash
python PnPJDU.py --help
python PnPJDU.py
```

You can also point to explicit folders if needed:

```bash
python PnPJDU.py --dataset_dir ./dataset --kernel_dir ./Kernels --output_dir ./Output
```

## License

This project is provided for research and educational use. Please cite the paper if you use this code or reproduce results.

## Citation

If you use this code in your work, please cite:

```bibtex
@article{layazali2026joint,
  title={Joint Hyperspectral Image Deconvolution and Unmixing via Plug-and-Play Priors},
  author={Layazali, Sina and Preza, Chrysanthe},
  journal={Remote Sensing},
  volume={18},
  number={13},
  pages={2066},
  year={2026},
  publisher={MDPI}
}

```
