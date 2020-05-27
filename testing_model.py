from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil

class ImageClassificationTest:
	def set_up_data_transform(self):
		"""Sets up the data transforms for the testing data"""
		self.data_transforms = {
		    'testing': transforms.Compose([
		        transforms.Resize(256),
		        transforms.CenterCrop(224),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		    ]),
		}

	def image_save(self, inp, file_name):
		"""Saves the image to the wrong images folder."""
		inp = inp.numpy().transpose((1, 2, 0))
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		inp = std * inp + mean
		inp = np.clip(inp, 0, 1)

		save_location = os.path.join("wrong_images", file_name)
		plt.imsave(save_location, inp)

	def save_misclassified_image(self, index):
		"""Saves the misclassified image and prints out what is was"""
		pred_class = self.classes[preds]
		act_class = self.classes[labels.data]
		file, _ = self.dataloaders['testing'].dataset.samples[i]

		print(f"Predicted {pred_class}")
		print(f"Actual {act_class}")
		print(f"File: {file}")
		print()

		shutil.copy(file, "wrong_images")
		self.image_save(inputs.cpu().data[0], f"pred_{pred_class}_act_{act_class}_{file[-12:-4]}.png")


	def test_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
		"""Tests the model chosen against the dataset"""
		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0
		model.eval()   # Set model to 'testinguate mode

		running_loss = 0.0
		running_corrects = 0

		for i, (inputs, labels) in enumerate(self.dataloaders['testing'], 0):
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)

			if (labels.data != preds):
				self.save_misclassified_image(i)

			# statistics
			running_corrects += torch.sum(preds == labels.data)

		acc = running_corrects.double() / self.dataset_sizes['testing']
		print("Acc: {:.2f}%".format(acc*100))


	def set_up_images(self):
		"""Gets the data from the folders specified and sets 
		it up to be used as training data"""
		image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
	                                  self.data_transforms[x])
	          for x in ['testing']}
		self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x])
		              for x in ['testing']}
		self.dataset_sizes = {x: len(image_datasets[x]) for x in ['testing']}

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def __init__(self, classes, data_dir, model_path):
		"""Saves necessary attributes and sets up the images
		with their data transforms"""
		self.classes = classes
		self.data_dir = data_dir
		self.model_path = model_path
		self.set_up_data_transform()
		self.set_up_images()		

	def set_up_test(self):
		"""Gets the model and calls the test method"""
		model_ft = torch.load(self.model_path)
		criterion = nn.CrossEntropyLoss()
		optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

		self.test_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
		                       num_epochs=1)

def main():
	classifier = ImageClassificationTest(["bird", "parakeet"], "Feeder11_bird_parakeet/", "best_bird_para_model11.pt")
	classifier.set_up_test()

main()
	