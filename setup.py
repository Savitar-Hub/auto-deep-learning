from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')

except (IOError, ImportError):
    long_description = open('README.md').read()


# Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]

# For extra parts of the packages
extras = {
    'cloud': ['pytest'],
    'api': ['fastapi', 'alembic']
}


setup(
    name='auto_deep_learning',
    version='0.1.5.3',
    description='Automation of the creation of the architecture of the neural network based on the input',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Nil-Andreu/auto-deep-learning',
    author='Nil Andreu',
    author_email='nilandreug@email.com',
    # data_files=[('', ['README.md'])],
    keywords=[
        'deep learning',
        'machine learning',
        'computer vision',
        'convolutional neural networks',
        'neural networks',
        'image classification'
    ],
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'torchaudio==0.13.1',
        'transformers==4.25.1',
        'sentence-transformers==2.2.2',
        'torchsummary==1.5.1',
        'pytest',
        'pre-commit==2.21.0',
        'pandas==1.5.0'
    ],
    extras_require=extras,      # For handling API & Cloud extras
    classifiers=classifiers,
)
