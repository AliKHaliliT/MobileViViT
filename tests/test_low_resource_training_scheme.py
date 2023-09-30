import unittest
import tensorflow as tf
from MobileViViT.assets.utils.low_resource_training_scheme import low_resource_training_scheme as lrts
import numpy as np


class TestProgressBar(unittest.TestCase):

    def test_model_wrong__type_type__error(self):

        # Arrange
        model = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=model, optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1)


    def test_optimizer_wrong__type_type__error(self):

        # Arrange
        optimizer = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=optimizer, 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1)
            

    def test_loss__fn_wrong__type_type__error(self):

        # Arrange
        loss_fn = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=loss_fn, 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1)


    def test_train__dataset_wrong__type_type__error(self):

        # Arrange
        train_dataset = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=train_dataset, 
                 epochs=1)
            

    def test_epochs_wrong__type_type__error(self):

        # Arrange
        epochs = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=epochs)
            
    
    def test_epochs_wrong__value_value__error(self):

        # Arrange
        epochs = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=epochs)
            

    def test_verbose_wrong__type_type__error(self):

        # Arrange
        verbose = None

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1, verbose=verbose)
            

    def test_verbose_wrong__value_value__error(self):

        # Arrange
        verbose = -1

        # Act and Assert
        with self.assertRaises(ValueError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1, verbose=verbose)
            

    def test_metrics_wrong__type_type__error(self):

        # Arrange
        metrics = ()

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1, metrics=metrics)


    def test_metrics_wrong__type__instances_type__error(self):

        # Arrange
        metrics = [tf.keras.metrics.Accuracy, tf.keras.metrics.AUC, None]

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1, metrics=metrics)
            

    def test_val__dataset_wrong__type_type__error(self):

        # Arrange
        val_dataset = 1

        # Act and Assert
        with self.assertRaises(TypeError):
            lrts(model=tf.keras.models.Sequential(), optimizer=tf.keras.optimizers.Adam(), 
                 loss_fn=tf.keras.losses.MeanSquaredError(), 
                 train_dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1, 2, 3])), 
                 epochs=1, val_dataset=val_dataset)


    def test_output_progress__parameters_str(self):

        # Arrange
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [(tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")), 
                   (tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"))]

        num_samples = 10
        input_dim = 10
        output_dim = 10
        num_epochs = 5
        batch_size = 5

        x_train = np.random.rand(num_samples, input_dim).astype(np.float32)
        y_train = np.random.randint(0, output_dim, size=num_samples).astype(np.int64)
        X_val = np.random.rand(num_samples, input_dim).astype(np.float32)
        y_val = np.random.randint(0, output_dim, size=num_samples).astype(np.int64)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Act
        output = lrts(model=model, optimizer=optimizer, loss_fn=loss_fn, 
                      train_dataset=train_dataset, epochs=num_epochs, 
                      metrics=metrics, val_dataset=val_dataset)

        # Assert
        self.assertTrue(isinstance(output, dict))


if __name__ == "__main__":
    unittest.main()