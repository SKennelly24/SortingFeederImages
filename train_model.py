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

class ImageClassifier:

	def load_trim_cnn(self):
		"""Loads the ResNet18 CNN pretrained and returns the model and
		everything else needed to fine tune the model"""

		model_ft = models.resnet18(pretrained=True)
		num_ftrs = model_ft.fc.in_features
		# Setting the amount of classes
		model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))

		model_ft = model_ft.to(self.device)

		criterion = nn.CrossEntropyLoss()

		# Observe that all parameters are being optimized
		optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

		# Decay LR by a factor of 0.1 every 7 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

		return model_ft, criterion, optimizer_ft, exp_lr_scheduler
		

	def fine_tune_model_train(self):
		"""Loads the model and trains it"""
		model_ft, criterion, optimizer_ft, exp_lr_scheduler = self.load_trim_cnn()
		model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, self.num_epochs)


	def train_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
	    since = time.time()

	    best_model_wts = copy.deepcopy(model.state_dict())
	    best_acc = 0.0

	    for epoch in range(num_epochs):
	        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	        print('-' * 10)

	        # Each epoch has a training and validation phase
	        for phase in ['train', 'val']:
	            if phase == 'train':
	                model.train()  # Set model to training mode
	            else:
	                model.eval()   # Set model to evaluate mode

	            running_loss = 0.0
	            running_corrects = 0

	            # Iterate over data.
	            for inputs, labels in self.dataloaders[phase]:
	                inputs = inputs.to(self.device)
	                labels = labels.to(self.device)

	                # zero the parameter gradients
	                optimizer.zero_grad()

	                # forward
	                # track history if only in train
	                with torch.set_grad_enabled(phase == 'train'):
	                    outputs = model(inputs)
	                    _, preds = torch.max(outputs, 1)
	                    loss = criterion(outputs, labels)

	                    # backward + optimize only if in training phase
	                    if phase == 'train':
	                        loss.backward()
	                        optimizer.step()

	                # statistics
	                running_loss += loss.item() * inputs.size(0)
	                running_corrects += torch.sum(preds == labels.data)
	            if phase == 'train':
	                scheduler.step()

	            epoch_loss = running_loss / self.dataset_sizes[phase]
	            epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

	            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
	                phase, epoch_loss, epoch_acc))

	            # deep copy the model
	            if phase == 'val' and epoch_acc > best_acc:
	                best_acc = epoch_acc
	                best_model_wts = copy.deepcopy(model.state_dict())

	        print()

	    time_elapsed = time.time() - since
	    print('Training complete in {:.0f}m {:.0f}s'.format(
	        time_elapsed // 60, time_elapsed % 60))
	    print('Best val Acc: {:4f}'.format(best_acc))

	    # load best model weights
	    model.load_state_dict(best_model_wts)
	    torch.save(model, self.save_location)
	    #print(model.state_dict())
	    return model


	def set_up_data_transforms(self):
		"""Sets up the data transforms that the training and testing
		data undergo"""
		self.data_transforms = {
		    'train': transforms.Compose([
		        transforms.RandomResizedCrop(224), #This or the next two lines were used
		        #transforms.Resize(256),
		        #transforms.CenterCrop(224),
		        transforms.RandomHorizontalFlip(),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		    ]),
		    'val': transforms.Compose([
		        transforms.Resize(256),
		        transforms.CenterCrop(224),
		        transforms.ToTensor(),
		        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		    ]),
		}

	def __init__(self, data_dir, save_location, epochs=40):
		"""Does the data transforms, sets up some attributes, gets the 
		classes and dataset"""
		self.set_up_data_transforms()

		self.num_epochs = epochs
		self.save_location = save_location

		image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
		                                          self.data_transforms[x])
		                  for x in ['train', 'val']}
		self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
		                                             shuffle=True, num_workers=4)
		              for x in ['train', 'val']}
		self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
		self.class_names = image_datasets['train'].classes
		print(self.class_names)

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
	classifier = ImageClassifier('Feeder11_bird_parakeet/training/training_data', "best_bird_para_model11.pt", 25)
	classifier.fine_tune_model_train()
	
main()


