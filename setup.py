from setuptools import setup

setup(
    name='auto_deep_learning',
    version='0.1',
    description='Automation of the creation of the architecture of the neural network based on the input',
    url='https://github.com/Nil-Andreu/auto-deep-learning',
    author='Nil Andreu',
    author_email='nilandreug@email.com',
    keywords=[
        'deep learning',
        'machine learning',
        'convolutional neural networks',
        'neural networks'
    ],
    license='MIT',
    packages=[
        'auto_deep_learning',
        'auto_deep_learning.test'
    ],
    zip_safe=False,
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'torchaudio==0.13.1',
        'pytest'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
  ],
)