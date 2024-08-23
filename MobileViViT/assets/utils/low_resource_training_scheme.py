import tensorflow as tf
from typing import Union, Optional
import sys
from .progress_bar import progress_bar


def low_resource_training_scheme(model: tf.keras.Model, 
                                 optimizer: tf.keras.optimizers.Optimizer, 
                                 loss_fn: tf.keras.losses.Loss,  
                                 train_dataset: Union[tf.data.Dataset, tf.keras.utils.Sequence],
                                 epochs: int,
                                 verbose: int = 1,
                                 metrics: Optional[list[tf.keras.metrics.Metric]] = None,
                                 val_dataset: Optional[Union[tf.data.Dataset, tf.keras.utils.Sequence]] = None) \
                                -> dict[str, list[Union[int, float]]]:

    """

    This method defines a custom training scheme for low-resource training.
    It does so by accumulating the gradients of the model depending on the batch size 
    and then applying them at the end of the epoch.
    When dealing with higher dimensional data, particularly when hardware resources are limited, 
    using large batches may not be feasible. In such cases, setting the batch size to one 
    and accumulating gradients could mitigate, to some extent, the negative impact of training
    with a single sample per batch.


    Parameters
    ----------
    model : tf.keras.Model
        Model to train.

    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to use.

    loss_fn : tf.keras.losses.Loss
        Loss function to use.
        
    train_dataset : tf.data.Dataset or tf.keras.utils.Sequence
        Training dataset.

    epochs : int
        Number of epochs to train the model.

    verbose : int, optional
        Verbosity mode. The default value is 1.
            The options are:
                0
                    Silent mode.
                1
                    Print the progress bar.
                2
                    Print the progress bar and the loss and metrics.

    metrics : list, optional
        List of metrics to use. The default value is None.

    val_dataset : tf.data.Dataset or tf.keras.utils.Sequence, optional
        Evaluation dataset. The default value is None.


    Returns
    -------
    history : dict
        Dictionary containing the training history.

    """

    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be a tf.keras.Model")
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise TypeError("optimizer must be a tf.keras.optimizers.Optimizer")
    if not isinstance(loss_fn, tf.keras.losses.Loss):
        raise TypeError("loss_fn must be a tf.keras.losses.Loss")
    if not isinstance(train_dataset, tf.data.Dataset) and not isinstance(train_dataset, tf.keras.utils.Sequence):
        raise TypeError("train_dataset must be a tf.data.Dataset or a tf.keras.utils.Sequence")
    if not isinstance(epochs, int):
        raise TypeError("epochs must be an int")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not isinstance(verbose, int):
        raise TypeError("verbose must be an int")
    if verbose not in [0, 1, 2]:
        raise ValueError("verbose must be 0, 1, or 2")
    if metrics is not None:
        if not isinstance(metrics, list):
            raise TypeError("metrics must be a list")
        for metric in metrics:
            if not isinstance(metric, tf.keras.metrics.Metric):
                raise TypeError("metrics must contain only tf.keras.metrics.Metric")
    if val_dataset is not None:
        if not isinstance(val_dataset, tf.data.Dataset):
            raise TypeError("val_dataset must be a tf.data.Dataset or a tf.keras.utils.Sequence")
    

    # Defining the loss metric
    loss_metric = tf.keras.metrics.Mean()

    # Defining the history dictionary
    history = {"loss": []}
    if metrics:
        for metric in metrics:
            history[metric.name] = []
    if val_dataset:
        history["val_loss"] = []
        if metrics:
            for metric in metrics:
                history[f"val_{metric.name}"] = []
    
    for epoch in range(epochs):

        # Initializing the total gradients to zero which will reset at the end of each epoch
        total_gradients = [tf.zeros_like(var) for var in model.trainable_variables]

        if verbose == 1 or verbose == 2:
            print(f"Epoch {epoch + 1}:")

        # Iterating over the training dataset
        for batch, (x_batch, y_batch) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(x_batch)
                loss = loss_fn(y_batch, predictions)

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            # Accumulating the gradients
            total_gradients = [acc_grad + grad for acc_grad, grad in zip(total_gradients, gradients)]


            # Updating the loss metric
            loss_metric(loss)

            # Updating the metrics
            if metrics:
                for metric in metrics:
                    metric(y_batch, predictions)


            if verbose == 1 or verbose == 2:

                # Printing the progress bar
                sys.stdout.write("\r" + progress_bar(batch + 1, len(train_dataset), description="Training: "))
                sys.stdout.flush()
        else:
            if verbose == 1:
                print('\t')


        # Applying the gradients
        optimizer.apply_gradients(zip(total_gradients, model.trainable_variables))


        # Appending the loss to the history dictionary
        history["loss"].append(loss_metric.result().numpy())

        # Appending the metrics to the history dictionary
        if metrics:
            for metric in metrics:
                history[metric.name].append(metric.result().numpy())


        if verbose == 2:
            # Printing the loss and metrics
            output_string = f" | loss: {loss_metric.result()}, "

            if metrics:

                for metric in metrics:
                    output_string += f"{metric.name}: {metric.result()}, "

                print(output_string[:-2])
            else:
                print(output_string[:-2])


        # Resetting the loss metric and the metrics
        loss_metric.reset_states()

        if metrics:
            for metric in metrics:
                metric.reset_states()


        # Evaluating the model on the validation dataset
        # The comments are the same as above
        if val_dataset:

            for batch, (x_batch, y_batch) in enumerate(val_dataset):

                predictions = model(x_batch)
                loss = loss_fn(y_batch, predictions)


                loss_metric(loss)

                if metrics:
                    for metric in metrics:
                        metric(y_batch, predictions)


                if verbose == 1 or verbose == 2:
                    
                    sys.stdout.write("\r" + progress_bar(batch + 1, len(val_dataset), description="Validation: "))
                    sys.stdout.flush()

            else:
                if verbose == 1:
                    print('\t')


            history["val_loss"].append(loss_metric.result().numpy())

            if metrics:
                for metric in metrics:
                    history[f"val_{metric.name}"].append(metric.result().numpy())


            if verbose == 2:
                output_string = " | "

                if metrics:

                    for metric in metrics:
                        output_string += f"val_{metric.name}: {metric.result()}, "

                    print(output_string[:-2])
                else:
                    print(output_string[:-2] + f" | val_loss: {loss_metric.result()}")


            loss_metric.reset_states()

            if metrics:
                for metric in metrics:
                    metric.reset_states()

    
    # Returning the history dictionary
    return history