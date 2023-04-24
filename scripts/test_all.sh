#
# Generate all testing artifacts from pre-trained models (steg and suds)
#    

# RQ1/RQ2 images
time python RQ1-RQ2-imgs.py

# RQ1/RQ2 stats
time python RQ1-RQ2-stats.py

# RQ3 plots
time python RQ3-plots.py

# RQ4-plots.py
time python RQ4-plots.py

# RQ5 images
time python RQ5-imgs.py

# combine results into figure 3
time python make_pretty_pictures.py

# CASE STUDY: DATA POISONING (312.51s)
time python examples/data_poison/test_data_poison.py