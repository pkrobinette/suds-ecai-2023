# SUDS: Sanitizing Universal and Dependent Steganography

## Installation
```
docker build -t suds_image . 
```

```
sudo docker run --name suds --rm -it suds_image bash
```


## Artifact Instructions
All models are pre-trained. Reproduce results by:

```
chmod +x scripts/*
./scripts/test_all.sh
```
2. If you would like to reproduce a specific figure, see the index below and run:
```
python *.py
```

## Results Index
All results are saved to the `results` folder. The results are indexable by research question or figure number.

- **Table 3:** `results/RQ1-RQ2-stats/all_img_stats.txt`
- **Figure 3a:** `results/suds-pretty-picture.pdf`
- **Figure 3b:** `results/noise-pretty-picture.pdf`
- **Figure 3c:** `results/cifar-suds-pretty-picture.pdf`
- **Figure 4:** `results/RQ3-plots/zsize_results.pdf`
- **Figure 5:** `results/RQ4-plots/latent_mappings_plot_compact.pdf`
- **Table 4:** `results/data_poison/classification_results.txt`



## File Descriptions
#### Training
> `suds_main.py`: main suds training file (sanitizing)

> `custom_main_deephide.py`: main udh and ddh training file (hiding)


#### Testing
> `RQ1`: SUDS ability to sanitize

> `RQ2`: SUDS comparison to Gaussian noise

> `RQ3`: flexibility; SUDS with different latent space dimensions

> `RQ4`: SUDS for detection

> `RQ5`: scalability; SUDS on CIFAR-10

> `CASE STUDY`: Data poisoning

#### Directories
> `configs`: training config files for udh and ddh

> `examples`: case studies: data poisoning

> `models`: all pre-trained models

> `results`: where results are saved

> `scripts`: easy to run training and testing scripts

> `utils`: Helper functions. Model files. Etc.

## Notes
- If docker throws a disk space error, try running this prior to building:
```
docker system prune
```

