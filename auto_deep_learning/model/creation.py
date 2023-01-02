from typing import Optional

import numpy as np
import torch
from torchsummary import summary

from auto_deep_learning.enum import ModelName
from auto_deep_learning.model.definition import define_model
from auto_deep_learning.model.inference import inference
from auto_deep_learning.utils import DatasetSampler
from auto_deep_learning.utils.config import ConfigurationObject
from auto_deep_learning.utils.functions import count_model_parameters, to_cuda
from auto_deep_learning.utils.model import (default_weight_init, get_criterion,
                                            get_optimizer)

conf_obj = ConfigurationObject()


# TODO: Save also the whole model (both in gpu and cpu) and load the whole model (without predefinition of the arch)
class Model:
    def __init__(
        self,
        data: DatasetSampler,
        category_type: Optional[str] = '',
        objective: Optional[str] = conf_obj.objective,
        model_name: Optional[ModelName] = None,
        model_version: Optional[str] = '',
        input_shape: Optional[int] = conf_obj.image_size,
    ):
        """Instance of the Neural Network model.

        Args:
            data (Loader): the loader of the data that will be used
            category_type (Optional[str], optional): short description of which task do you want to do. Defaults to None.
            objective (Optional[ModelObjective], optional): definition of which is the objective.
                Defaults to throughput, as are the simpler and optimized for production environments.
            model_name (Optional[ModelName], optional): definition of which is the model name (from HuggingFace).
            model_version (Optional[str], optional): definition of which is the model version (from HuggingFace).
        """

        self.data = data
        self.objective = objective
        self.model_name = model_name
        self.category_type = category_type
        self.model_version = model_version
        self.input_shape = input_shape

        self.model = define_model(
            data=self.data,
            category_type=self.category_type,
            objective=self.objective,
            model_name=self.model_name,
            model_version=self.model_version,
            input_shape=self.input_shape,  # TODO: Adapt to different input shapes
        )

        self.criterion = get_criterion()

    @classmethod
    def fit(
        cls,
        lr: Optional[
            int
        ],  # TODO: Create function for default lr  -> 1e-4? Depends on self.model.recommended_lr, self.model.recommended_n_epochs
        n_epochs: Optional[
            int
        ] = conf_obj.n_epochs,  # TODO: Create function for default lr
        use_cuda: Optional[bool] = torch.cuda.is_available(),
        save_path: Optional[str] = 'model.pt',
    ):
        """Train of the model

        Args:
            lr (int): the learning rate of the optimizer
            n_epochs (int): number of epochs that we will train
            use_cuda (bool, optional): whether we train the model in cuda. Defaults to torch.cuda.is_available().
            save_path (str, optional): the path to save the best model. Defaults to 'model.pt'.

        Returns:
            model: the model trained

        """

        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        optimizer = get_optimizer(cls.model, lr)

        if use_cuda:
            cls.model.cuda()

        for epoch in range(1, n_epochs + 1):
            # initialize variables to monitor training and validation loss
            train_loss, valid_loss = 0.0, 0.0

            # set the module to training mode
            cls.model.train()

            for batch_idx, (data, target) in enumerate(cls.data['train']):
                # move to GPU
                if use_cuda:
                    # TODO: Multiple targets and data (for now data is only img)
                    data, target = to_cuda(data), to_cuda(target)

                optimizer.zero_grad()
                # Obtain the output from the model
                output = cls.model(
                    data
                )  # TODO: Multiple outputs, and each of them needs to be compared to the target

                # TODO: Check this works fine
                # Obtain loss for each of the targets we have
                for target_key in target.keys():
                    target_output = output[target_key]
                    target_expected = target[target_key]

                    if loss in locals():
                        loss += cls.criterion(target_output, target_expected)

                    else:
                        loss = cls.criterion(target_output, target_expected)

                # Backward induction
                loss.backward()
                # Perform optimization step
                optimizer.step()

                train_loss = train_loss + (
                    (1 / (batch_idx + 1)) * (loss.data.item() - train_loss)
                )
                del loss

            # set the model to evaluation mode
            cls.model.eval()

            if valid_loader := cls.data.get('valid'):
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # move to GPU
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()

                    output = cls.model(data)

                    # Obtain the loss
                    for target_key in target.keys():
                        target_output, target_expected = (
                            output[target_key],
                            target[target_key],
                        )

                        if loss in locals():
                            loss += cls.criterion(target_output, target_expected)

                        else:
                            loss = cls.criterion(target_output, target_expected)

                    # Add this loss to the list (same as before but instead of train we use valid)
                    valid_loss = valid_loss + (
                        (1 / (batch_idx + 1)) * (loss.data.item() - valid_loss)
                    )
                    del loss

                # TODO: Use logger instead of prints
                # print training/validation statistics
                print(
                    'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                        epoch, train_loss, valid_loss
                    )
                )

                if valid_loss < valid_loss_min:
                    # Print an alert
                    print(
                        'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model..'.format(
                            valid_loss_min, valid_loss
                        )
                    )

                    torch.save(cls.model.state_dict(), save_path)

                    # Update the new minimum
                    valid_loss_min = valid_loss

        return cls.model

    @classmethod
    def test(cls, use_cuda: bool = torch.cuda.is_available()):
        # monitor test loss and accuracy
        test_loss = 0.0
        correct = 0.0
        total = 0.0

        # set the module to evaluation mode
        cls.model.eval()

        # Move cuda after eval so consumes less memory
        if use_cuda:
            cls.model.cuda()

        if loader_test := cls.data.get('test'):
            for batch_idx, (data, target) in enumerate(loader_test):
                # move to GPU
                if use_cuda:
                    data, target = to_cuda(data), to_cuda(target)

                # forward pass: compute predicted outputs by passing inputs to the model
                output = cls.model(data)

                # calculate the loss
                # TODO: Testing with multiple outputs & targets
                loss = cls.criterion(output, target)

                # update average test loss
                test_loss = test_loss + (
                    (1 / (batch_idx + 1)) * (loss.data.item() - test_loss)
                )

                # convert output probabilities to predicted class
                pred = output.data.max(1, keepdim=True)[1]

                # compare predictions to true label
                correct += np.sum(
                    np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()
                )

                total += data.size(0)

        print('Test Loss: {:.6f}\n'.format(test_loss))

        print(
            '\nTest Accuracy: %2d%% (%2d/%2d)'
            % (100.0 * correct / total, correct, total)
        )

    @classmethod
    def reset(cls):

        cls.model = default_weight_init(cls.model)

    @classmethod
    def predict(cls, img_path: str = 'predict.img'):
        output_inference = inference(cls.model, img_path=img_path)

        return output_inference

    @classmethod
    def model_parameters(cls):
        # TODO: Use log instead
        print(summary(cls.model, (cls.input_shape)))

    @property
    def count_parameters(self):
        return count_model_parameters(self.model)
