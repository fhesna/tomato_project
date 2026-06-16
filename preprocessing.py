import tensorflow as tf
from tensorflow.keras import layers

def get_datasets(train_dir, val_dir, image_size, batch_size, model_type):

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

    if model_type == 'MobileNetV2':
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_type == 'EfficientNetB1':
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    elif model_type == 'ResNet50':
        preprocess_input = tf.keras.applications.resnet.preprocess_input
    else:
        raise ValueError("Unknown model type")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y)
    )

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y)
    )

    return train_ds, val_ds
