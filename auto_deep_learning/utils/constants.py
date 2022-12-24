# For batch processing
BATCH_SIZE = 64

# Train/Valid/Test Split
TEST_SIZE = 0.05
VALID_SIZE = 0.1

# For image transformation
MEAN_CONSTANTS = [0.485, 0.456, 0.406]
STD_CONSTANTS = [0.229, 0.224, 0.225]

# TODO: Singleton for storing the constants, so they can be changed accross the project