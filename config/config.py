# Encoder configuration
encoder = {
    'input_shape':(256,256,3),
    'conv_layers':3,
    'filter_conv_layer':[64,128,128,64],
    'kernel_size_conv_layer':[3,3,3,3],
    'activation_conv_layer':['relu','relu','relu','relu'],
    'stride_conv_layer':[2,2,2,2],
    'padding_conv_layer':['same','same','same','same'],
    'dense_layers':2,
    'node_dense_layer':[300,300],
    'activation_dense_layer':['relu','relu'],
    'latent_dim':200,

}

# Decoder configuration
decoder = {
    'latent_dim':200,
    'reshape_layer':(16,16,64),
    'conv_transpose_layers':4,
    'filter_conv_transpose_layers':[64,128,128,32],
    'kernel_size_conv_layer':[3,3,3,3],
    'activation_conv_layer':['relu','relu','relu','relu'],
    'stride_conv_layer':[2,2,2,2],
    'padding_conv_layer':['same','same','same','same'],

}

# Variational autoencoder
vae_optimizer = 'adam'

# Train configuration
train_size = 10

train = {
    'epochs':10,
    'batch_size':2,

}


# Conv2D configuration
def conv2d_config(
    filters = 64,
    kernel_size = 3,
    activation='relu',
    strides = 2,
    padding = 'same',

):
    return {
    'filters':filters,
    'kernel_size':kernel_size,
    'activation' : activation,
    'strides' : strides,
    'padding' : padding,

    }

# Dense configuration
def dense_config(
    units = 300,
    activation='relu',

):
    return {
    'units':units,
    'activation': activation,

    }

# Conv2DTranspose configuration
def conv2dtranspose_config(
    filters = 64,
    kernel_size = 3,
    activation='relu',
    strides = 2,
    padding = 'same',

):
    return {
    'filters':filters,
    'kernel_size':kernel_size,
    'activation' : activation,
    'strides' : strides,
    'padding' : padding,

    }
