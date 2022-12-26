class ModelObjective(Enum):
    """Define which is the model objective
    If you want to make research, it is recommended to use for accuracy.
    If you are creating a startup or is for a company project, where speed is important, use throughput.

    Args:
        Enum
    """

    accuracy = 'accuracy'
    throughput = 'throughput' 