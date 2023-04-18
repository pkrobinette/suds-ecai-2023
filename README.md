# SUDS: Sanitizing Universal and Dependent Steganography

## Installation

## Artifact Instructions
1. The models must be trained before the results can be replicated.
```
chmod +x scripts/*
./scripts/train_all.sh
```
2. Replicate the results
```
./scripts/test_all.sh
```
## Results Index
All results are saved to the `results` folder. The results are indexeable by research question or figure number.

- **RQ1:** Images -> `noise_vs_suds_demo`, Stats -> `noise_comparison`
- **RQ2:** Images -> `noise_vs_suds_demo`, Stats -> `noise_comparison`
- **RQ3:** `feature_size_img_stats`
- **RQ4:** `latent_mappings`
- **RQ5:** `sanitize_demo_cifar`
---
- **Table 3:** `noise_comparison/all_img_stats.txt`
- **Figure 3a:** `noise_vs_suds_demo/suds-pretty-picture.pdf`
- **Figure 3b:** `noise_vs_suds_demo/noise-pretty-picture.pdf`
- **Figure 3c:** `sanitize_demo_cifar/cifar-suds-pretty-picture.pdf`
- **Figure 4:** `feature_size_img_stats/z_size_results.pdf`
- **Figure 5:** `latent_mappings/latent_mappings_plot_compact.pdf`
- **Table 4:** `data_poison/classification_results.txt`



## File Descriptions
#### Training
> `suds_main.py`: main suds training file (sanitizing)

> `dhide_main.py`: main udh and ddh training file (hiding)


#### Testing
> `generate_steg_demo.py`: generate images demonstrating lsb hiding, ddh hiding, and udh hiding.

> `generate_suds_demo.py`: generate images demonstrating sanitization peformance of suds.

> `generate_noise_demo.py`: generate image demonstrating the ability of noise to sanitize steg images.

> `generate_noise_stats.py`: generate image quality stats for all sanitization techniques and steg hide functions.

> `generate_zsize_results.py`: compare suds performance for different latent space sizes (z)

> `generate_latent_mapping.py`: explore where covers and containers get mapped to in the latent space.

#### Directories
> `configs`: training config files for udh and ddh

> `data`: where mnist and cifar are loaded and stored

> `examples`: case studies: data poisoning, stegomalware

> `models`: all pre-trained models

> `results`: where results are saved

> `scripts`: easy to run training and testing scripts

> `utils`: the brains of the operation. Helper functions. Model files. Etc.

## Notes
For questions, please feel free to contact me.
