import bayesflow as bf
import keras


class CNN(bf.networks.SummaryNetwork):
    def __init__(
        self,
        summary_dim,
        conv_params,
        num_fully_connected=15,
        conv_dropout_prob=0.0,
        dense_dropout_prob=0.0,
        kernel_regularizer="l1l2",
        **kwargs,
    ):
        """
        Create a pattern summary network with flexible convolutional layers, max-pooling, and dropout.

        Parameters
        ----------
        summary_dim : int
            The dimension of the summary statistics.
        conv_params : list of dict
            A list of dictionaries defining the parameters for convolutional layers, e.g.:
                - num_filters: int, number of filters for the Conv2D layer.
                - kernel_size: int or tuple, size of the kernel for the Conv2D layer.
                - pool_size: int or tuple, size of the max-pooling window (optional).
                - activation: str, activation function (default is 'relu').
        num_fully_connected : int, optional
            The number of units in the fully connected layer (default is 15).
        conv_dropout_prob : float, optional
            The dropout probability applied after each convolutional layer (default is 0.0).
        dense_dropout_prob : float, optional
            The dropout probability applied before the fully connected layer (default is 0.0).
        kernel_regularizer : string or keras.Regularizer object
            Regularization of convolutional and dense layers (default is "l1l2")
        kwargs : dict, optional
            Additional keyword arguments for the tf.keras.Model superclass.
        """
        super().__init__(**kwargs)
        self.summary_dim = summary_dim

        self.net = keras.models.Sequential()
        self.net.add(keras.layers.Lambda(lambda x: keras.ops.expand_dims(x, axis=-1)))

        for conv_param in conv_params:
            num_filters = conv_param.get("num_filters", 32)
            kernel_size = conv_param.get("kernel_size", 3)
            activation = conv_param.get("activation", "relu")
            pool_size = conv_param.get("pool_size", None)

            self.net.add(
                keras.layers.Conv2D(
                    num_filters,
                    kernel_size,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                )
            )

            if pool_size is not None:
                self.net.add(keras.layers.MaxPooling2D(pool_size))

            if conv_dropout_prob > 0:
                self.net.add(keras.layers.Dropout(conv_dropout_prob))

        self.net.add(keras.layers.Flatten())
        if dense_dropout_prob > 0:
            self.net.add(keras.layers.Dropout(dense_dropout_prob))
        self.net.add(
            keras.layers.Dense(
                num_fully_connected,
                activation="relu",
                kernel_regularizer=kernel_regularizer,
            )
        )
        self.net.add(keras.layers.Dense(summary_dim, kernel_regularizer=kernel_regularizer))

    def call(self, inputs, training: bool = False, **kwargs):
        return self.net(inputs)
