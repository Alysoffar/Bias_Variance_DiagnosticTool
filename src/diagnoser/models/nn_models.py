import tensorflow as tf

def build_nn_classifier(config, input_dim):

    hidenn_layers = config.get("hidden_layers", [64, 32])
    l2 = config.get("l2",0.01)
    lr = config.get("learning_rate", 0.001)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))

    for units in hidenn_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2)))
        
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
    