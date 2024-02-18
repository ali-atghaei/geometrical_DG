# Geometrical_DG
Domain Generalization via Geometric Adaptation over Augmented Data

This repo provides a demo for our paper "Domain Generalization via Geometric Adaptation over Augmented Data" on the PACS or any other mentioned dataset.

## Requirements 

python 3.8 

pytorch 1.1 

## Test 

For running the evaluation code, please download the PACS dataset from <a href='http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017'>pacs download</a>. Then update the files with suffix _test.txt in data/datalists for each domain, following styles below:
<code>
  /home/user/data/images/PACS/art_painting/dog/pic_001.jpg 0
  /home/user/data/images/PACS/art_painting/dog/pic_002.jpg 0
  /home/user/data/images/PACS/art_painting/dog/pic_003.jpg 0
</code>

Just write the name of target domain in shell_test.py file and just run it.


## Train from scratch 
In data/DGDataLoader.py set the available_datasets =  pacs_dataset.
In data/datalists for each domain, you should have a suffix_train.txt and suffix_val.txt and suffix_test.txt like the section above. 
In the shell_train.py file write the domain names of dataset and write the target domain name. 
after that just run the shell_train.py 

##Acknowledgements

The core of our code is sourced from the repositories listed below.

<a href='https://github.com/KaiyangZhou/DG-research-pytorch'>DDAIG</a> "Domain Generalization by Solving Jigsaw Puzzles", CVPR 2019 paper. 

<a href='https://github.com/MediaBrain-SJTU/FACT'>FACT</a>  "A Fourier-based Framework for Domain Generalization", CVPR 2021 paper. 

We express our gratitude to the authors for making their codes publicly available. We also encourage acknowledging their contributions by citing their publications.

