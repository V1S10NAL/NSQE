"""
SE_ResUNet.py - Network architectures for Reconstruction and Classification
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import CosineSimilarity
import tools

def se_block(inputs, filters, ratio=8):
    """SE Block"""
    se = layers.GlobalAveragePooling1D()(inputs)
    se = layers.Dense(units=filters // ratio, activation='relu')(se)
    se = layers.Dense(units=filters, activation='sigmoid')(se)
    out = layers.multiply([inputs, se])
    return out

def residual_block(input_tensor, num_filters, kernel_size=3):
    """Residual block without SE"""
    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)


    if input_tensor.shape[-1] != num_filters:
        shortcut = layers.Conv1D(num_filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters, kernel_size=3):
    """Encoder block"""
    x = residual_block(input_tensor, num_filters, kernel_size=kernel_size)
    p = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters, kernel_size=3):
    """Decoder block"""
    x = layers.Conv1DTranspose(num_filters, kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = residual_block(x, num_filters, kernel_size=kernel_size)
    return x

def se_residual_block(input_tensor, num_filters, kernel_size=3):
    """Residual block with SE"""
    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = se_block(x, num_filters)

    if input_tensor.shape[-1] != num_filters:
        shortcut = layers.Conv1D(num_filters, kernel_size=1, padding='same')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def se_encoder_block(input_tensor, num_filters, kernel_size=3):
    """Encoder block with SE"""
    x = se_residual_block(input_tensor, num_filters, kernel_size=kernel_size)
    p = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x, p

def se_decoder_block(input_tensor, skip_tensor, num_filters, kernel_size=3):
    """Decoder block with SE"""
    x = layers.Conv1DTranspose(num_filters, kernel_size=3, strides=2, padding='same')(input_tensor)
    x = layers.concatenate([x, skip_tensor])
    x = se_residual_block(x, num_filters, kernel_size=kernel_size)
    return x

def ResUNet(num_features, num_outputs, args):
    """ResUNet for Spectral Reconstruction"""
    inputs = layers.Input(shape=(num_features, 1))

    num_filters = 64
    x1, p1 = encoder_block(inputs, num_filters)
    x2, p2 = encoder_block(p1, num_filters * 2)
    x3, p3 = encoder_block(p2, num_filters * 4)
    x4, p4 = encoder_block(p3, num_filters * 8)
    dropout_rate = 0.1
    p4 = layers.Dropout(dropout_rate)(p4)

    b1 = residual_block(p4, num_filters * 16)

    d1 = decoder_block(b1, x4, num_filters * 8)
    d2 = decoder_block(d1, x3, num_filters * 4)
    d3 = decoder_block(d2, x2, num_filters * 2)
    d4 = decoder_block(d3, x1, num_filters)

    outputs = layers.Conv1D(num_outputs, kernel_size=1, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-5))(d4)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.9772,
        staircase=False
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=adam_optimizer,
                  loss='mean_squared_error',
                  metrics=['mae', 'mape', 'mse', CosineSimilarity(axis=-1), tools.LogCoshError()])
    model.summary()
    return model

def seResNet_classification(num_features, num_outputs, args):
    """SE-ResNet for Classification"""
    inputs = layers.Input(shape=(num_features, 1))

    x1, p1 = se_encoder_block(inputs, 64)
    x2, p2 = se_encoder_block(p1, 128)
    x3, p3 = se_encoder_block(p2, 256)
    x4, p4 = se_encoder_block(p3, 512)

    c = layers.Conv1D(512, kernel_size=1, activation='relu', name='classification_branch')(p4)
    c_avg = layers.GlobalAveragePooling1D()(c)
    c_max = layers.GlobalMaxPooling1D()(c)
    c = layers.Concatenate()([c_avg, c_max])

    c = layers.Dense(256, activation='relu')(c)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(0.05)(c)
    c = layers.Dense(64, activation='softmax')(c)
    c = layers.Dropout(0.05)(c)
    c = layers.Dense(args.num_substances, activation='softmax', name='classification')(c)

    model = tf.keras.Model(inputs=inputs, outputs=c)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=args.num_spectra * 0.7 / args.batch_size,
        decay_rate=0.9772,
        staircase=False
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    model.compile(optimizer=adam_optimizer,
                  loss={'classification': 'categorical_crossentropy'},
                  loss_weights={'classification': 1.0},
                  metrics={'classification': ['categorical_accuracy',
                                              tf.keras.metrics.Precision(name='precision'),
                                              tf.keras.metrics.Recall(name='recall')]})
    model.summary()
    return model