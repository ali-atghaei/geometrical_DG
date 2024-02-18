import argparse
import ast
from collections import deque
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from models.model_factory import *
from optimizer.optimizer_helper import get_optim_and_scheduler
from data import *
from utils.Logger import Logger
from utils.tools import *
import copy
from scipy.sparse import csgraph
import warnings

# from loss import MaximalCodingRateReduction

warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
	parser.add_argument("--target", choices=available_datasets, help="Target")
	parser.add_argument("--input_dir", default=None, help="The directory of dataset lists")
	parser.add_argument("--output_dir", default=None, help="The directory to save logs and models")
	parser.add_argument("--config", default=None, help="Experiment configs")
	parser.add_argument("--tf_logger", type=ast.literal_eval, default=False, help="If true will save tensorboard compatible logs")

	args = parser.parse_args()
	config_file = "config." + args.config.replace("/", ".")
	print(f"\nLoading experiment {args.config}\n")
	config = __import__(config_file, fromlist=[""]).config

	return args, config


class Trainer:
	def __init__(self, args, config, device):
		self.args = args
		self.config = config
		self.device = device
		self.global_step = 0

		# networks
		self.encoder = get_encoder_from_config(self.config["networks"]["encoder"]).to(self.device)
		self.classifier = get_classifier_from_config(self.config["networks"]["classifier"]).to(self.device)
		
		# optimizers
		self.encoder_optim, self.encoder_sched = \
			get_optim_and_scheduler(self.encoder, self.config["optimizer"]["encoder_optimizer"])
		self.classifier_optim, self.classifier_sched = \
			get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
				# dataloaders
		self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
		self.val_loader = get_val_dataloader(args=self.args, config=self.config)
		self.test_loader = get_test_loader(args=self.args, config=self.config)
		self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}
	

	def guassian_similarity_matrix_1(self,matrix,sigma=1.0):
		pairwise_distance = torch.cdist(matrix,matrix)
		similarity = torch.exp(-pairwise_distance.pow(2)/(2*sigma**2))
		#optional
		# similarity = similarity*(1-torch.eye(matrix.size(0)))
		return similarity
	
	def gaussian_similarity_matrix_2(self,matrix1, matrix2, sigma=1.0):
		# Compute pairwise Euclidean distances
		pairwise_distances = torch.cdist(matrix1, matrix2, p=2.0)
		# Compute Gaussian similarity using the RBF kernel
		similarity_matrix = torch.exp(-pairwise_distances.pow(2) / (2 * sigma**2))

		return similarity_matrix
	
	
			
	def laplacian_matrix(self,adj_matrix):
		degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
		laplacian = degree_matrix - adj_matrix
		return laplacian

	
	def manifold_distance(self,adj_matrix1, adj_matrix2, Lst):
		laplacian_s = self.laplacian_matrix(adj_matrix1)
		# print ("lap=",laplacian_s)
		laplacian_t = self.laplacian_matrix(adj_matrix2)
		laplacian_s_t = self.laplacian_matrix(Lst)
		epsilon = 1e-6
		laplacian_s = laplacian_s + (epsilon * torch.eye(laplacian_s.size(0))).to(self.device)
		laplacian_t = laplacian_t + epsilon * torch.eye(laplacian_t.size(0)).to(self.device)
		laplacian_s_t = laplacian_s_t + epsilon * torch.eye(laplacian_s_t.size(0)).to(self.device)

		eigenvalues_s, eigenvectors_s = torch.symeig(laplacian_s, eigenvectors=True)
		eigenvalues_t, eigenvectors_t = torch.symeig(laplacian_t, eigenvectors=True)
		diagonal_matrix_eigenvalues_s = torch.diag(eigenvalues_s)
		diagonal_matrix_eigenvalues_t = torch.diag(eigenvalues_t)

		##############
		# Check if the matrix is singular
		if any(diagonal_matrix_eigenvalues_t.diag() == 0):
			# Perturb the diagonal elements with a small positive constant
			epsilon = 1e-6
			diagonal_matrix_eigenvalues_t = diagonal_matrix_eigenvalues_t + (torch.eye(diagonal_matrix_eigenvalues_t.size(0)) * epsilon).to(self.device)

			# Now, the matrix should be invertible
			inverse_matrix_t = torch.inverse(diagonal_matrix_eigenvalues_t)
			
		else:
			# The matrix is already invertible
			inverse_matrix_t = torch.inverse(diagonal_matrix_eigenvalues_t)
		if any(diagonal_matrix_eigenvalues_s.diag() == 0):
			# Perturb the diagonal elements with a small positive constant
			epsilon = 1e-6
			diagonal_matrix_eigenvalues_s = diagonal_matrix_eigenvalues_s + (torch.eye(diagonal_matrix_eigenvalues_s.size(0)) * epsilon).to(self.device)

			# Now, the matrix should be invertible
			inverse_matrix_s = torch.inverse(diagonal_matrix_eigenvalues_s)
			
		else:
			# The matrix is already invertible
			inverse_matrix_s = torch.inverse(diagonal_matrix_eigenvalues_s)
			
		##############
		approximated_eigenvector_S = laplacian_s_t @ eigenvectors_t @ inverse_matrix_t
		approximated_source_matrix = approximated_eigenvector_S @ diagonal_matrix_eigenvalues_t @ approximated_eigenvector_S.t() #eq10article
		difference = approximated_source_matrix - laplacian_s
		frobenius_norm = torch.norm(difference, p='fro')

		return frobenius_norm#.item()


	def exp_cosine_similarity_matrix(self,features):
		# Normalize the feature vectors
		features_normalized = F.normalize(features, p=2, dim=1)

		# Compute pairwise cosine similarity
		cosine_similarity_matrix = torch.mm(features_normalized, features_normalized.t())

		# Apply exponential function element-wise
		exp_cosine_similarity_matrix = torch.exp(cosine_similarity_matrix)

		return exp_cosine_similarity_matrix
	def entropy_last(self,tensor):
		# Calculate entropy for a probability distribution
		return -torch.sum(tensor * torch.log2(tensor + 1e-10))

	def calculate_entropy_last(self,features, labels):
		# Create a dictionary to store features for each label
		grouped_features = {}

		# Iterate over features and labels to group them
		for feature, label in zip(features, labels):
			label = int(label.item())  # Convert label to integer
			if label not in grouped_features:
				grouped_features[label] = []
			grouped_features[label].append(feature)

		# Calculate entropy for each group
		entropies = []
		for label, feature_group in grouped_features.items():
			feature_tensor = torch.stack(feature_group)
			# feature_tensor = feature_tensor.view(-1) #flattened
			probabilities = F.softmax(feature_tensor, dim=1)#1
			feature_entropies = -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=0)
			entropies.append(feature_entropies)
			

		# Sum the entropies
		sum_entropies = torch.sum(torch.stack(entropies))
		mean_entropies = torch.mean(torch.stack(entropies))
		return sum_entropies , mean_entropies
	def entropy(self,tensor):
		# Move the tensor to GPU if available
		# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		tensor = tensor.to(self.device)
		
		# Flatten the tensor and convert it to a 1D tensor
		flattened_tensor = tensor.view(-1)
		
		# Calculate histogram on GPU using bincount
		hist = torch.bincount(flattened_tensor.long(), minlength=256)
		
		# Compute the probability distribution of values in the tensor
		probs = hist.float() / flattened_tensor.numel()
		
		# Calculate entropy
		entropy_val = -torch.sum(probs * torch.log2(probs + 1e-10))  # Adding a small epsilon to avoid log(0)
		
		return entropy_val
	def calculate_entropy(self,features, labels):
		# Create a dictionary to store features for each label
		grouped_features = {}

		# Iterate over features and labels to group them
		for feature, label in zip(features, labels):
			label = int(label.item())  # Convert label to integer
			if label not in grouped_features:
				grouped_features[label] = []
			grouped_features[label].append(feature)

		# Calculate entropy for each group
		# entropies = []
		sum_entropies = 0
		for label, feature_group in grouped_features.items():
			feature_tensor = torch.stack(feature_group)
			# feature_tensor = feature_tensor.view(-1) #flattened
			ent = self.entropy(feature_tensor)
			sum_entropies+=ent
			# entropies.append(feature_entropies)
			

		# Sum the entropies
		
		mean_entropies = sum_entropies / feature.size(0)
		return sum_entropies , mean_entropies

	def _do_epoch(self):
		criterion = nn.CrossEntropyLoss()

		# turn on train mode
		self.encoder.train()
		self.classifier.train()
		# self.encoder_teacher.train()
		# self.classifier_teacher.train()
		# self.classifier_middle.train()

		for it, (batch, label, domain) in enumerate(self.train_loader):
			
			# print ("batch[4]",batch[4].shape)
			label_ori = torch.cat([label[0],label[1]])
			
			# print ("label",label)

			# preprocessing
			batch = torch.cat(batch, dim=0).to(self.device)
			label = torch.cat(label, dim=0).to(self.device)
			
			# zero grad
			self.encoder_optim.zero_grad()
			self.classifier_optim.zero_grad()
			# self.classifier_middle_optim.zero_grad()

			# forward
			loss_dict = {}
			correct_dict = {}
			num_samples_dict = {}
			total_loss = 0.0

			features = self.encoder(batch)
			scores = self.classifier(features)
			

			assert batch.size(0) % 2 == 0
			split_idx = int(batch.size(0) / 2)
			
			scores_ori, scores_aug = torch.split(scores, split_idx)
			feat_ori , feat_aug = torch.split(features, split_idx)
			sigma_guassian = 10
			################################################
			total_similarity_matrix  = self.guassian_similarity_matrix_1(features,sigma_guassian)
			
			L_total_sim = self.laplacian_matrix(total_similarity_matrix)
			mask_same_labels = torch.zeros_like(L_total_sim)
			# Create masks for each label
			for l in torch.unique(label):
				# Create a mask for the current label
				mask = (label.unsqueeze(1) == l) & (label.unsqueeze(0) == l)
				
				mask_same_labels+=mask
			
			mask_different_labels = 1-mask_same_labels
			L_same_labels = mask_same_labels * L_total_sim
			L_different_labels = mask_different_labels * L_total_sim
			frobenius_norm_same = torch.norm(L_same_labels, p='fro')
			similarity_of_same_class = 10*frobenius_norm_same / torch.sum(mask_same_labels) 
			frobenius_norm_dif = torch.norm(L_different_labels, p='fro')
			similarity_of_different_class = 10*frobenius_norm_dif / torch.sum(mask_different_labels) 
			
			################################################
			
			feature_entropies = torch.sum(features * torch.log2(features + 1e-20), dim=0) #ghablan manfi bud 
			mean_entropy_all= torch.mean(feature_entropies)
			
			K_S_ori_sim = self.guassian_similarity_matrix_1(feat_ori, sigma_guassian)
			norm_adj_ori = torch.norm(K_S_ori_sim, p='fro')
			
			
			K_S_aug_sim = self.guassian_similarity_matrix_1(feat_aug, sigma_guassian)
			norm_adj_aug = torch.norm(K_S_aug_sim, p='fro')
			
			L_st = self.gaussian_similarity_matrix_2(feat_ori,feat_aug,sigma_guassian)
			
			difference = self.manifold_distance(K_S_ori_sim, K_S_aug_sim, L_st)
		
			labels_ori, labels_aug = torch.split(label, split_idx)

			
			assert scores_ori.size(0) == scores_aug.size(0)

			# classification loss for original data
			loss_cls = criterion(scores_ori, labels_ori)
			loss_dict["main"] = loss_cls.item()
			correct_dict["main"] = calculate_correct(scores_ori, labels_ori)
			num_samples_dict["main"] = int(scores.size(0) / 2)

			# classification loss for augmented data
			loss_aug = criterion(scores_aug, labels_aug)
			loss_dict["aug"] = loss_aug.item()
			correct_dict["aug"] = calculate_correct(scores_aug, labels_aug)
			num_samples_dict["aug"] = int(scores.size(0) / 2)

			#scale to become betweem 0 and 1 
			diff_show = difference
			sim_same = similarity_of_same_class
			sim_dif = similarity_of_different_class
			
			const_weight = get_current_consistency_weight(epoch=self.current_epoch,
														  weight=self.config["lam_const"],
														  rampup_length=self.config["warmup_epoch"],
														  rampup_type=self.config["warmup_type"])
			cont_loss = 5 *similarity_of_different_class - 1*similarity_of_same_class 
			
			total_loss = 0.9 * loss_cls + 0.9 * loss_aug + 0.25*difference +\
							0.1*cont_loss + 0.01*mean_entropy_all  
							
			
			loss_dict['mandif']=difference

			loss_dict['similarity_same_cls'] = similarity_of_same_class

			loss_dict['similarity_diff_cls'] = similarity_of_different_class

			loss_dict['frob_feat'] = mean_entropy_all

			
			loss_dict["total"] = total_loss.item()
			# backward
			
			total_loss.backward()
			

			# update
			self.encoder_optim.step()
			self.classifier_optim.step()
			# self.classifier_middle_optim.step()
			self.global_step += 1

			

			# record
			self.logger.log(
				it=it,
				iters=len(self.train_loader),
				losses=loss_dict,
				samples_right=correct_dict,
				total_samples=num_samples_dict
			)

		# turn on eval mode
		self.encoder.eval()
		self.classifier.eval()
		

		# evaluation
		with torch.no_grad():
			for phase, loader in self.eval_loader.items():
				total = len(loader.dataset)
				class_correct = self.do_eval(loader)
				class_acc = float(class_correct) / total
				self.logger.log_test(phase, {'class': class_acc})
				self.results[phase][self.current_epoch] = class_acc

			# save from best val
			if self.results['val'][self.current_epoch] >= self.best_val_acc:
				self.best_val_acc = self.results['val'][self.current_epoch]
				self.best_val_epoch = self.current_epoch + 1
				self.logger.save_best_model(self.encoder, self.classifier, self.best_val_acc)

	def do_eval(self, loader):
		correct = 0
		for it, (batch, domain) in enumerate(loader):
			data, labels, domains = batch[0].to(self.device), batch[1].to(self.device), domain.to(self.device)
			features = self.encoder(data)
			scores = self.classifier(features)
			correct += calculate_correct(scores, labels)
		return correct


	def do_training(self):
		self.logger = Logger(self.args, self.config, update_frequency=100)
		self.logger.save_config()

		self.epochs = self.config["epoch"]
		self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

		self.best_val_acc = 0
		self.best_val_epoch = 0

		for self.current_epoch in range(self.epochs):

			# step schedulers
			self.encoder_sched.step()
			self.classifier_sched.step()

			self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
			self._do_epoch()
			self.logger.finish_epoch()

		# save from best val
		val_res = self.results['val']
		test_res = self.results['test']
		self.logger.save_best_acc(val_res, test_res, self.best_val_acc, self.best_val_epoch - 1)

		return self.logger


def main():
	args, config = get_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trainer = Trainer(args, config, device)
	trainer.do_training()


if __name__ == "__main__":
	torch.backends.cudnn.benchmark = True
	main()
