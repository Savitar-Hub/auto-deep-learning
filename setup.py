from setuptools import setup

setup(
    name='auto_nn',
    version='0.1',
    description='Automation of the creation of the architecture of the neural network based on the input',
    url='https://github.com/Nil-Andreu/auto-nn',
    author='Nil Andreu',
    author_email='nilandreug@email.com',
    license='MIT',
    packages=[
        'auto_nn',
        'auto_nn.test'
    ],
    zip_safe=False,
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'torchaudio==0.13.1',
        'pytest'
    ]
)