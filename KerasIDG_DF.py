import tensorflow as tf
import pandas as pd

def image_ImageDataGenerator_from_dataframe(model, normalize_method, target_size, idg_params, class_mode,
                                            epochs, train_test_split=0.2, df=None, test_df=None, train_path=None,
                                            test_path=None, x_col=None, y_col=None):
    """ImageDataGenerator from DataFrame

    This function trains a Keras model using ImageDataGenerator.flow_from_dataframe(). Note that IDG cannot be used on TPUs.
    Arguments:
        model: a tf.keras.Model object
                A tf.keras.Model object. input_shape must be declared and model must be compiled first
        train_path: str of path, default None
            training data path. If `x_col` in DataFrame is the absolute path, no need to define this
        test_path: str of path, default None
            test data path. If `x_col` in DataFrame is the absolute path, no need to define this
        normalize_method: dict
            Dict contains normalization method, as describe in:
                https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#arguments_11.
            Sample usage: {"featurewise_center": True}, {"featurewise_std_normalization" :True}
            Note: does not support featurewise params. Use samplewise instead.
        target_size: tuple of 2
            A tuple of (img_width, img_height).
        idg_params: dict
            dictionary contains ImageDataGenerator params. It should not contain one of the following: samplewise_center,
                samplewise_std_normalization, rescale as it is the `normalize_method`.
            Sample usage:
            {
            "rotation_range" = 30,
            "horizontal_flip"=True,
            "vertical_flip"=True
            }
        class_mode: str or None
            class mode of ImageDataGenerator. Should be one of "binary", "categorical", "sparse" or None
        epochs: int
            number of epochs to train
        train_test_split: float, between 0 and 1
            Split ratio between train and test set. For example if 0.2, the training set has 80% and validation set has 20%

        df: pandas DataFrame, default None
                Dataframe of training set. Must be defined if `from_directory` is False
        test_df: pandas DataFrame, default None
            DataFrame of test set. Must be defined if `from_directory` is False
        x_col: str, default None
            Column of images' names in DataFrame. If it is absolute path, train_directory can be None. Train DF and Test DF should have the same x_col.
        y_col: str, default None
            Column of images' labels in DataFrame. Train DF and Test DF should have the same y_col.
    Returns:
        model: a tf.keras.Model
            trained Keras model
        history: a History object
            History object of model after training
        predictions: numpy.array(s)
            numpy.array(s) of predictions after training
    """
    idg_train = tf.keras.preprocessing.image.ImageDataGenerator(
        **normalize_method,
        **idg_params,
        validation_split=train_test_split,
    )
    idg_test = tf.keras.preprocessing.image.ImageDataGenerator(**normalize_method)
    train_ds = idg_train.flow_from_dataframe(
        dataframe=df,
        directory=train_path,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=32,
        class_mode=class_mode,
        subset="training"
    )
    valid_ds = idg_train.flow_from_dataframe(
        dataframe=df,
        directory=train_path,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=32,
        class_mode=class_mode,
        subset="validation"
    )
    test_ds = idg_test.flow_from_dataframe(
        dataframe=test_df,
        directory=test_path,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        class_mode=None,
    )

    history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[
        tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.EarlyStopping(patience=epochs // 10, restore_best_weights=True)
    ])
    predictions = model.predict(test_ds)
    return model, history, predictions