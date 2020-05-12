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
		model_ft = models.resnet18(pretrained=True)
		num_ftrs = model_ft.fc.in_features
		# Here the size of each output sample is set to 2.
		# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
		model_ft.fc = nn.Linear(num_ftrs, 2)

		model_ft = model_ft.to(self.device)

		criterion = nn.CrossEntropyLoss()

		# Observe that all parameters are being optimized
		optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

		# Decay LR by a factor of 0.1 every 7 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

		return model_ft, criterion, optimizer_ft, exp_lr_scheduler
		

	def fine_tune_model_train(self):
		#Load and trim data
		model_ft, criterion, optimizer_ft, exp_lr_scheduler = self.load_trim_cnn()
		model_ft = self.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

		self.visualize_model(model_ft)

	def load_trim_final_layer(self):
		model_conv = torchvision.models.resnet18(pretrained=True)
		for param in model_conv.parameters():
		    param.requires_grad = False

		# Parameters of newly constructed modules have requires_grad=True by default
		num_ftrs = model_conv.fc.in_features
		model_conv.fc = nn.Linear(num_ftrs, 2)

		model_conv = model_conv.to(self.device)

		criterion = nn.CrossEntropyLoss()

		# Observe that only parameters of final layer are being optimized as
		# opposed to before.
		optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

		# Decay LR by a factor of 0.1 every 7 epochs
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
		return model_conv, criterion, optimizer_conv, exp_lr_scheduler

	def convnet_feature_extraction_train(self):
		model_conv, criterion, optimizer_conv, exp_lr_scheduler = self.load_trim_final_layer()
		model_conv = self.train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
		self.visualize_model(model_conv)

	def imshow(self, inp, title=None):
		"""Imshow for Tensor."""
		inp = inp.numpy().transpose((1, 2, 0))
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		inp = std * inp + mean
		inp = np.clip(inp, 0, 1)
		if title is not None:
			plt.title(title)
		plt.pause(0.001)  # pause a bit so that plots are updated
		plt.imsave("figure.png", inp)

	def showDataChanges(self):
		# Get a batch of training data
		inputs, classes = next(iter(self.dataloaders['train']))

		# Make a grid from batch
		out = torchvision.utils.make_grid(inputs)

		self.imshow(out, title=[self.class_names[x] for x in classes])

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
	    return model

	def visualize_model(self, model, num_images=6):
	    was_training = model.training
	    model.eval()
	    images_so_far = 0
	    fig = plt.figure()

	    with torch.no_grad():
	        for i, (inputs, labels) in enumerate(self.dataloaders['val']):
	            inputs = inputs.to(self.device)
	            labels = labels.to(self.device)

	            outputs = model(inputs)
	            _, preds = torch.max(outputs, 1)

	            for j in range(inputs.size()[0]):
	                images_so_far += 1
	                ax = plt.subplot(num_images//2, 2, images_so_far)
	                ax.axis('off')
	                ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
	                self.imshow(inputs.cpu().data[j])

	                if images_so_far == num_images:
	                    model.train(mode=was_training)
	                    return
	        model.train(mode=was_training)

	def set_up_data_transforms(self):
		self.data_transforms = {
		    'train': transforms.Compose([
		        transforms.RandomResizedCrop(224),
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

	def __init__(self):
		# Data augmentation and normalization for training
		# Just normalization for validation
		self.set_up_data_transforms()

		data_dir = 'data/hymenoptera_data'
		image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
		                                          self.data_transforms[x])
		                  for x in ['train', 'val']}
		self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
		                                             shuffle=True, num_workers=4)
		              for x in ['train', 'val']}
		self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
		self.class_names = image_datasets['train'].classes

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
	classifier = ImageClassifier()
	classifier.showDataChanges()
	#classifier.fine_tune_model_train()
	classifier.convnet_feature_extraction_train()
	
main()


