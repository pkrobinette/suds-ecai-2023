#!/bin/bash
#
# Generate all testing artifacts from pre-trained models (steg and suds)
#    

# RQ1/RQ2 images
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (1/7): RQ1-RQ2-imgs ***"
echo -e "****************************************\n\n"
python RQ1-RQ2-imgs.py

# RQ1/RQ2 stats
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (2/7): RQ1-RQ2-stats ***"
echo -e "****************************************\n\n"
python RQ1-RQ2-stats.py

# RQ3 plots
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (3/7): RQ3-plots ***"
echo -e "****************************************\n\n"
python RQ3-plots.py

# RQ4-plots.py
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (4/7): RQ4-plots ***"
echo -e "****************************************\n\n"
python RQ4-plots.py

# RQ5 images
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (5/7): RQ5-imgs ***"
echo -e "****************************************\n\n"
python RQ5-imgs.py

# combine results into figure 3
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (6/7): Combine images ***"
echo -e "****************************************\n\n"
python make_pretty_pictures.py

# CASE STUDY: DATA POISONING (312.51s)
echo -e "\n\n****************************************"
echo "*** RUNNING TEST (7/7): Data Poison ***"
echo -e "****************************************\n\n"
python examples/data_poison/test_data_poison.py