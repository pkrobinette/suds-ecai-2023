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

All results are saved to the `results` folder. Descriptions of each folder and their corresponding research questions are below.

- RQ1: 

- RQ2

- RQ3

- RQ4

- RQ5

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
