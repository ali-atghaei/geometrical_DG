# geometrical_DG
Domain Generalization via Geometric Adaptation over Augmented Data

This repo provides a demo for our paper "Domain Generalization via Geometric Adaptation over Augmented Data" on the PACS or any other mentioned dataset.

#requirement 

python 3.8 

pytorch 1.1 

For running the evaluation code, please download the PACS dataset from <a href='http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017'>pacs download</a>. Then update the files with suffix _test.txt in data/datalists for each domain, following styles below:
<code>
  /home/user/data/images/PACS/art_painting/dog/pic_001.jpg 0
  /home/user/data/images/PACS/art_painting/dog/pic_002.jpg 0
  /home/user/data/images/PACS/art_painting/dog/pic_003.jpg 0
</code>

#Train from scratch 

In the shell_train.py file write the domain names of dataset and write the target domain name. 
after that just run the shell_train.py 

#Test 

Just write the name of target domain in shell_test.py file and just run it. 


