from auto_deep_learning.model import Model


def default_weight_init(
    model: Model
):

    """Function to reset the parameters of the model

    Args:
        m (Model): the model that we want to reset
    """

    # We find the method
    reset_parameters = getattr(model, 'reset_parameters', None)

    # If the method is callable
    if callable(reset_parameters):

        # We call it
        model.reset_parameters()
    
    return model