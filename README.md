# Auto-Deep-Learning (Auto Deep Learning)
[![Downloads](https://static.pepy.tech/personalized-badge/auto-deep-learning?period=month&units=none&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/auto_deep_learning) ![Version](https://img.shields.io/badge/version-0.1.1-blue) ![Python-Version](https://img.shields.io/badge/python-3.9-blue) ![issues](https://img.shields.io/github/issues/Nil-Andreu/auto_deep_learning) ![PyPI - Status](https://img.shields.io/pypi/status/auto_deep_learning) ![License](https://img.shields.io/github/license/Nil-Andreu/auto_deep_learning) 

```auto_deep_learning```: with this package, you will be able to create, train and deploy neural networks automatically based on the input that you provide.

## Alert
This package is still on development, but Start the Project to know further updates in next days!
For the moment, would be for computer vision classification tasks on images (multi-modal included).

## Installation
Use the package manager [pip](https://pypi.org/project/pip/) to install *auto_deep_learning*.

To install the package:
```bash
    pip install auto_deep_learning
```

**If using an old version of the package, update it:**
```bash
    pip install --upgrade auto_deep_learning
```


## Basic Usage
How easy can be to create and train a deep learning model:
```python
    from auto_deep_learning import Model
    from auto_deep_learning.utils import Loader, image_folder_convertion

    df = image_folder_convertion()
    data = Loader(df)
    model = Model(data)
    model.fit()
    model.predict('image.jpg')
```


### Dataset

The data that it expects is a pd.DataFrame(), where the columns are the following:
```
    - image_path: the path to the image
    - class1: the classification of the class nr. 1. For example: {t-shirt, glasses, ...}
    - class2: the classification of the class nr. 2. For example: {summer, winter, ...}
    - ...
    - split_type: whether it is for training/validation/testing
```
For better performance, it is suggested that the classes and the type are of dtype *category* in the pandas DataFrame.
If the type is not provided in the dataframe, you should use the utils function of *data_split_types* (in *utils.dataset.sampler* file). 

If instead you have the images ordered in the structure of ImageFolder, which is the following structure:
```
    train/  
        class1_value/
            1.jgp
            2.jpg
            ...
        class2_value/
            3.jpg
            4.jpg
            ...
    test/
        class1_value/
            1.jgp
            2.jpg
            ...
        class2_value/
            3.jpg
            4.jpg
            ...
```
For simplifying logic, we have provided a logic that gives you the expected dataframe that we wanted, with the function of *image_folder_convertion* (in *utils.functions*), where it is expecting a path to the parent folder where the *train/* and */test* folders are.