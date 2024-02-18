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
In <code>data/DGDataLoader.py</code> set the <code>available_datasets =  pacs_dataset</code>.
In <code>data/datalists</code> for each domain, you should have a <code>suffix_train.txt</code> and <code>suffix_val.txt</code> and <code>suffix_test.txt</code> like the section above. 
In the <code>shell_train.py</code> file write the domain names of dataset and write the target domain name. 
After that just run the <code>shell_train.py</code> 

## Acknowledgements

The core of our code is sourced from the repositories listed below.
<ul>
  <li><a href='https://github.com/KaiyangZhou/DG-research-pytorch'>DDAIG</a> "Domain Generalization by Solving Jigsaw Puzzles", CVPR 2019 paper.</li>
  <li><a href='https://github.com/MediaBrain-SJTU/FACT'>FACT</a>  "A Fourier-based Framework for Domain Generalization", CVPR 2021 paper. </li>
</ul>

We express our gratitude to the authors for making their codes publicly available. We also encourage acknowledging their contributions by citing their publications.

