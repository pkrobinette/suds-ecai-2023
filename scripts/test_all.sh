#
# Generate all testing artifacts from pre-trained models (steg and suds)
#    
# generate results shown in figure 5: steg performance
python generate_steg_demo.py

# generate results shown in figure 6: (Section 4.2) suds performance
python generate_suds_demo.py --dataset mnist

# generate results shown in table 3: (Section 4.3) baseline comparison table
python generate_img_stats.py

# generate results shown in figure 7: (Section 4.3) baseline comparison images
python generate_noise_demo.py

# generate results shown in table 4: (Section 5.1), using suds with stegomalware
./generate_stegomalware.sh

# generate figure 8, table 6: (Section 5.2) feature ablation study results
python generate_zsize_results.py

# generate figure 9: (Section 5.3) A closer look at the latent space
python generate_latent_mappings.py

# generate table 5: (Section 5.4) data poisoning
python examples/data_poison/test_data_poison.py

# generate figure 10 data: (Section 5.5) suds applied to color images
python generate_suds_demo.py --dataset cifar