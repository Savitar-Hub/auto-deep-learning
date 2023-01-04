from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class SimpleConvNet(nn.Module):
    def __init__(
            self,
            input_shape: Tuple[int],
            map_class_name_length: Dict[str, int]
    ):
        super(SimpleConvNet, self).__init__()

        self.map_class_name_length = map_class_name_length

        # Define layers of a CNN
        self.conv_1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        # And also define a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # And linear models
        self.x_shape: int = input_shape[0] / 2 ** 5
        self.y_shape: int = input_shape[1] / 2 ** 5

        self.fc1 = nn.Linear(
            self.x_shape * self.y_shape * 256, 1024
        )  # 7 as (224/(2*2*2*2*2) and 256 as it is the depth

        # Other model that i tested
        for class_name, class_length in self.map_class_name_length.items():
            setattr(self, f'fc2_{class_name}', nn.Linear(1024, 512))
            setattr(self, f'fc3_{class_name}', nn.Linear(512, 256))
            setattr(self, f'fc4_{class_name}', nn.Linear(256, class_length))

        # Will define a dropout of 0.3
        self.dropout = nn.Dropout(0.2)

    def forward(self, x) -> Dict[str, Any]:
        # Define forward behavior
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        x = self.pool(F.relu(self.conv_4(x)))
        x = self.pool(F.relu(self.conv_5(x)))

        # Flatten the input
        x = x.view(-1, self.x_shape * self.y_shape * 256)

        # Add the dropout
        x = self.dropout(x)

        # Add the linear layers
        forward_values = {}
        for class_name in self.map_class_name_length.keys():
            new_x = self.dropout(F.relu(self.fc1(x)))
            new_x = self.dropout(getattr(self, f'fc2_{class_name}')(new_x))
            new_x = self.dropout(getattr(self, f'fc3_{class_name}')(new_x))
            new_x = getattr(self, f'fc4_{class_name}')(new_x)

            forward_values[class_name] = new_x

        # The class groups should be ordered, so those values always refer to the same classes
        return forward_values
