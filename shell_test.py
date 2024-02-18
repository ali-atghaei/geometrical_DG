import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--domain", "-d", default="sketch", help="Target")
parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
parser.add_argument("--times", "-t", default=1, type=int, help="Repeat times")

args = parser.parse_args()

###############################################################################

# source = ['Art', 'Product', 'Clipart', 'Real_World']
source = ["photo", "cartoon", "art_painting", "sketch"]
# target = args.domain
target = 'cartoon'
# target = 'Real_World'
# target = 'Clipart'
# target = 'Art'
source.remove(target)

# input_dir = 'path/to/data'
# output_dir = 'path/to/output'
# ckpt_path = f'./ckpt/{target}/best_model.tar'


input_dir = 'data/datalists'
output_dir = 'output/log'

#cartoon 
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-02-23-33-37/best_model.tar'#plain
ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-02-20-23-00/best_model.tar'#with nystrom
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-02-17-36-13/best_model.tar'#with nystrom
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-03-10-51-28/best_model.tar'#with nystrom index10
ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-03-12-58-04/best_model.tar' #nystrom euclidean 0.33
ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-10-17-19-22/best_model.tar'
ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-10-16-12-10/best_model.tar'
ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/cartoon/2023-12-10-12-57-38/best_model.tar'
#product 
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/FACT-main/output/log/PACS_ResNet50/Product/2023-07-04-18-06-26/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-11-08-23-33-35/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-11-08-22-16-12/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-09-26-11-05-31/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-11-19-11-35-42/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-11-22-12-11-38/best_model.tar' #FACT main
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Product/2023-11-22-13-08-43/best_model.tar'

#real world

# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-08-15-42-52/best_model.tar'#0.7833
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-08-13-25-50/best_model.tar' #0.7905
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-07-13-45-43/best_model.tar'#0.7925
# ckpt_path  = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-08-15-42-52/best_model.tar'#0.7833
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-10-10-25-29/best_model.tar'#0.7542
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-10-19-56-59/best_model.tar'#0.7914
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-10-22-37-37/best_model.tar'#0.7939
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-15-15-53-54/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Real_World/2023-11-18-13-54-20/best_model.tar'

#Clipart
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Clipart/2023-11-08-19-47-35/best_model.tar' #57.98
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Clipart/2023-11-08-18-25-46/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Clipart/2023-11-07-14-58-00/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Clipart/2023-07-07-05-44-03/best_model.tar'

#Art
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-07-23-27-11/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-08-09-57-48/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-09-10-26-27/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-08-09-57-48/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-09-12-48-23/best_model.tar'
# ckpt_path = '/mnt/hard/home/atghaei/w/distillation/mehr3/edited/FACT-main/output/log/PACS_ResNet50/Art/2023-11-09-21-22-25/best_model.tar'
config = "PACS/ResNet50"


domain_name = target
path = os.path.join(output_dir, config.replace("/", "_"), domain_name)
##############################################################################

for i in range(args.times):
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} '
              f'python test.py '
              f'--source {source[0]} {source[1]} {source[2]} '
              f'--target {target} '
              f'--input_dir {input_dir} '
              f'--output_dir {output_dir} '
              f'--config {config} '
              f'--ckpt {ckpt_path}')
