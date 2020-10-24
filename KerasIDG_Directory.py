import tensorflow as tf


def image_ImageDataGenerator_from_directory(model, train_path, test_path, normalize_method, target_size, idg_params,
                                            class_mode, epochs, valid_path=None, train_test_split=0.2):
    """ImageDataGenerator from directory

    This function trains a Keras model using ImageDataGenerator.flow_from_directory(). Note that IDG cannot be used on TPUs.

    Arguments:
        model: a tf.keras.Model object
            A tf.keras.Model object. input_shape must be declared and model must be compiled first
        train_path: str of path
            training data path. The path should be as follows:
                --train
                ---classA
                ----img1.png
                ----img2.png
                ---classB
                ----img3.png
                ...
            Then: train_path = "../train"
        test_path: str of path
            test data path. The path should be as follows:
            --test
            ---test
            ----img10.png
            ----img11.png
            ...
            Then: train_path = "../test"
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
            "rotation_range":30,
            "horizontal_flip":True,
            "vertical_flip":True
            }
        class_mode: str or None
            class mode of ImageDataGenerator. Should be one of "binary", "categorical", "sparse" or None
        epochs: int
            number of epochs to train
        train_test_split: float, between 0 and 1
            Split ratio between train and test set. For example if 0.2, the training set has 80% and validation set has 20%

    Returns:
        model: a tf.keras.Model
            trained Keras model
        history: a History object
            History object of model after training
        predictions: numpy.array(s)
            numpy.array(s) of predictions after training
    """
    idg_test = tf.keras.preprocessing.image.ImageDataGenerator(**normalize_method)

    if not valid_path:
        idg_train = tf.keras.preprocessing.image.ImageDataGenerator(
            **normalize_method,
            **idg_params,
            validation_split=train_test_split,
        )

        train_ds = idg_train.flow_from_directory(
            train_path,
            target_size=target_size,
            class_mode=class_mode,
            subset="training"
        )
        valid_ds = idg_train.flow_from_directory(
            train_path,
            target_size=target_size,
            class_mode=class_mode,
            subset="validation"
        )
    else:
        idg_train = tf.keras.preprocessing.image.ImageDataGenerator(
            **normalize_method,
            **idg_params
        )
        idg_valid = tf.keras.preprocessing.image.ImageDataGenerator(
            **normalize_method,
            **idg_params
        )
        train_ds = idg_train.flow_from_directory(
            train_path,
            target_size=target_size,
            class_mode=class_mode
        )
        valid_ds = idg_valid.flow_from_directory(
            valid_path,
            target_size=target_size,
            class_mode=class_mode
        )

    test_ds = idg_test.flow_from_directory(test_path, target_size=target_size, class_mode=None)

    history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[
        tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.EarlyStopping(patience=epochs // 10, restore_best_weights=True)
    ])
    predictions = model.predict(test_ds)
    return model, history, predictions
