import tensorflow as tf
from gym.spaces import Box, Tuple

def observation_input(input_data, ob_space, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the 
    observation space type
    Params:
    
    ob_space: observation space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''
    if input_data is None:
        if isinstance(ob_space, Box):
            input_shape = (batch_size,) + ob_space.shape
            input_x = tf.placeholder(shape=input_shape, dtype=ob_space.dtype, name=name)
            processed_x = tf.to_float(input_x)
            return input_x, processed_x

        elif isinstance(ob_space, Tuple):
            assert all([isinstance(space, Box) for space in ob_space.spaces])
            input_x = tuple([tf.placeholder(shape=(batch_size,) + space.shape,
                                            dtype=space.dtype, name=name)
                             for space in ob_space.spaces])
            processed_x = tuple([tf.to_float(input) for input in input_x])
            return input_x, processed_x

        else:
            raise NotImplementedError
    else:
        if isinstance(ob_space, Box):
            return input_data.X, tf.to_float(input_data.X)

        elif isinstance(ob_space, Tuple):
            assert all([isinstance(space, Box) for space in ob_space.spaces])
            processed_x = [tf.to_float(x) for x in input_data.X] \
                if isinstance(input_data.X, tuple) else tf.to_float(input_data.X)
            return input_data.X, processed_x

        else:
            raise NotImplementedError

 
