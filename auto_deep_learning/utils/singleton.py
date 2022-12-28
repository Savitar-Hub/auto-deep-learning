class Singleton(type):
    _instances = {}

    # When we have a metaclass of a new class
    def __call__(cls, *args, **kwargs):
        # If this class is not in our instances
        if cls not in cls._instances:
            # Initialize a new and unique instance
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        
        # Return this class
        return cls._instances[cls]