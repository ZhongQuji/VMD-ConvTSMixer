import tensorflow as tf
from tensorflow.keras import layers

def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer with convolutional layers."""
    norm = layers.LayerNormalization if norm_type == 'L' else layers.BatchNormalization

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Conv1D(filters=x.shape[-1], kernel_size=3, padding='same', activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=3, padding='same', activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same')(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    return x + res

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
    """Build TSMixer model with convolutional layers."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
        x = x[:, :, target_slice]

    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel]

    return tf.keras.Model(inputs, outputs)


