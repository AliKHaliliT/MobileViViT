import tensorflow as tf
from typing import Union, Any


class SineLayerInitializer(tf.keras.initializers.Initializer):

    """

    Sine layer initializer from the paper "Implicit Neural Representations with Periodic Activation Functions"
    Link: https://arxiv.org/abs/2006.09661

    """

    def __init__(self, units: int, is_first: bool, omega_0: Union[int, float]) -> None:
    
        """

        Constructor of the Sine layer initializer.


        Parameters
        ----------
        units : int
            Number of input features.
        
        is_first : bool
            Whether the layer is the first layer of the network.

        omega_0 : int or float
            Scaling factor of the Sine activation function. Default value in the original implementation is 30.0.
            See paper Sec. 3.2., final paragraph, and supplement Sec. 1.5. for discussion of omega_0.

        
        Returns
        -------
        None.

        """
            
        if not isinstance(units, int):
            raise TypeError("units must be an integer")
        if units < 0:
            raise ValueError("units must be positive")
        if not isinstance(is_first, bool):
            raise TypeError("is_first must be a boolean")
        if not isinstance(omega_0, int) and not isinstance(omega_0, float):
            raise TypeError("Omega must be an integer or a float")

        
        super().__init__()
    
        self.units = units
        self.is_first = is_first
        self.omega_0 = omega_0


    def __call__(self, shape: tf.TensorShape, dtype: tf.DType = tf.dtypes.float32) -> tf.Tensor:

        """

        Call method of the Sine layer initializer.


        Parameters
        ----------
        shape : tf.TensorShape
            Shape of the tensor to initialize.

        dtype : tf.DType, optional
            Data type of the tensor to initialize.

            
        Returns
        -------
        tf.Tensor
            Initialized tensor.

        """

        if self.is_first:

            minval = -1 / self.units
            maxval = 1 / self.units

        else:

            minval = (- tf.sqrt(6 / self.units)) / (self.omega_0)
            maxval = (tf.sqrt(6 / self.units)) / (self.omega_0)


        return tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    

    def get_config(self) -> dict[str, Any]:

        """

        Method to get the configuration of the Sine layer initializer.


        Parameters
        ----------
        None.


        Returns
        -------
        config : dict
            Configuration of the Sine layer initializer.

        """
            
        config = super().get_config()

        config.update({
            "units": self.units,
            "is_first": self.is_first,
            "omega_0": self.omega_0
        })


        return config