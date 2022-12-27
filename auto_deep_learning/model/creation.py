"""
Things we take into consideration for creating the model:
- Complexity of the task: how many images and how many classes we want to infer
- Objective of the model: if a model if for production for a company, we might want to optimize tradeoff between throughtput and accuracy

The training of the model:
- Similarity with previous dataset: most of the models are trained with Imagenet and then we apply Transfer Learning. So depending on similarity, we make some warmup of the weights of the whole feature selection for some epochs or we only train the last layers.
- Type of the model: each model supported has the HP for which it was trained
"""
from typing import Optional

import torch
import numpy as np

from auto_deep_learning.enum import (
    ModelObjective,
    ModelName
)
from auto_deep_learning.utils import Loader
from auto_deep_learning.utils.model import get_criterion, get_optimizer
from auto_deep_learning.model.creation import define_model


class Model:
    def __init__(
        self,
        data: Loader,
        description: Optional[str] = '',
        objective: Optional[ModelObjective] = 'throughput',
        model_name: Optional[ModelName] = '',
        model_version: Optional[str] = ''
    ):
        """Instance of the Neural Network model.

        Args:
            data (Loader): the loader of the data that will be used
            description (Optional[str], optional): short description of which task do you want to do. Defaults to None.
            objective (Optional[ModelObjective], optional): definition of which is the objective. Defaults to throughput, as are the simpler and optimized for production environments.
            model_name (Optional[ModelName], optional): definition of which is the model name (from HuggingFace). 
            model_version (Optional[str], optional): definition of which is the model version (from HuggingFace).
        """
        
        self.data = data
        self.objective = objective
        self.model_name = model_name
        self.description = description
        self.model_version = model_version

        self.model = define_model(
            data=self.data,
            description=self.description,
            objective=self.objective,
            model_name=self.model_name,
            model_version=self.model_version
        )

        self.optimizer = get_optimizer(self.model)
        self.criterion = get_criterion()


    @classmethod
    def train(
        self,
        n_epochs,
        use_cuda: bool = torch.cuda.is_available(),
        save_path: str = 'model.pt'
    ):

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf 

        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            # set the module to training mode
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.data['train']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                # Obtain the output from the model
                output = self.model(data)
                # Obtain loss
                loss = self.criterion(output, target)
                # Backward induction
                loss.backward()
                # Perform optimization step
                self.optimizer.step()  

                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))

            # set the model to evaluation mode
            self.model.eval()

            if valid_loader := self.data.get('valid'):
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # move to GPU
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    # Obtain the loss
                    loss = self.criterion(output, target)
                    # Add this loss to the list (same as before but instead of train we use valid)
                    valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))

                # print training/validation statistics 
                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch, 
                    train_loss,
                    valid_loss
                    ))

                if valid_loss < valid_loss_min:
                    # Print an alert
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model..'.format(
                        valid_loss_min,
                        valid_loss))

                    torch.save(self.model.state_dict(), save_path)
                    
                    # Update the new minimum
                    valid_loss_min = valid_loss
            
        return self.model
        