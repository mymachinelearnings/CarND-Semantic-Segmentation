#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)

print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #loads graph definition and variables to the session provided, in this case 'sess'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    encoder_layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    #Scaling layers give better results as per the authors of VGG
#     encoder_layer_3 = tf.multiply(encoder_layer_3, 0.0001, name="pool3_out_scaled")
    
    encoder_layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
#     encoder_layer_4 = tf.multiply(encoder_layer_4, 0.0001, name="pool4_out_scaled")
    
    encoder_layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, encoder_layer_3, encoder_layer_4, encoder_layer_7

print('Unit testing load_vgg()')
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    """
    VGG16 downsampled using 5 pooling layers in total which means by 32 times
    In Decoder network, upsample 5 times, everytime by 2

    During Skip Layers which is essentially an element wise addition, you need to convolve the VGG layers to 1x1 to make the shapes same
    Here are the shapes obtained by printing
    >>> decoder_layer1 Shape ::  (?, ?, ?, 2)
    >>> vgg_layer4_out Shape ::  (?, ?, ?, 512)
    >>> vgg_layer4_1x1 Shape ::  (?, ?, ?, 2)

    """
    encoder_layer_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME', 
                                         kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    decoder_layer1 = tf.layers.conv2d_transpose(encoder_layer_1x1, num_classes, 4, 2, padding='SAME', 
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME', 
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                      kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    decoder_layer1_skip_add = tf.add(decoder_layer1, vgg_layer4_1x1)
#     print('>>> decoder_layer1 Shape :: ', decoder_layer1.shape)
#     print('>>> vgg_layer4_out Shape :: ', vgg_layer4_out.shape)
#     print('>>> vgg_layer4_1x1 Shape :: ', vgg_layer4_1x1.shape)

    
    decoder_layer2 = tf.layers.conv2d_transpose(decoder_layer1_skip_add, num_classes, 4, 2, padding='SAME', 
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME', 
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                      kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    decoder_layer2_skip_add = tf.add(decoder_layer2, vgg_layer3_1x1)
    
    decoder_layer3 = tf.layers.conv2d_transpose(decoder_layer2_skip_add, num_classes, 16, 8, padding='SAME', 
                                                kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                                kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))
    
    return decoder_layer3

print('Unit testing layers()')
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    """
    logits & labels need to be reshaped as the softmax function expects a 2D Tensor (?, ?, ?, 2)
    ValueError: Logits has wrong rank.  Tensor Variable/read:0 must have rank 2.  Received rank 4, shape (2, 3, 4, 2)
    
    before reshaping - logits (2, 3, 4, 2) labels (?, ?, ?, 2)
    After reshaping - logits (24, 2) labels (?, 2)
    """
    
#     print('before reshaping - logits {} labels {}'.format(nn_last_layer.shape, correct_label.shape))
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
#     print('After reshaping - logits {} labels {}'.format(logits.shape, labels.shape))
   
    cross_entropy_loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss_fn)
    
    return logits, train_op, cross_entropy_loss_fn

print('Unit testing optimize()')
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    
    print('Training Started, epochs {}'.format(epochs))
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch + 1))
        for image, label in get_batches_fn(batch_size):
#             print('image >> ', image)
#             print('label >> ', label)
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:image, correct_label:label, keep_prob:0.5, learning_rate:0.0001})
            print("Loss: = {:.3f}".format(loss))

print('Unit testing train_nn()')
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        
        epochs = 20
        batch_size = 5
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, encoder_layer_3, encoder_layer_4, encoder_layer_7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(encoder_layer_3, encoder_layer_4, encoder_layer_7, num_classes)
        
        # TODO: Train NN using the train_nn function
        logits, train_op, cross_entropy_loss_fn = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        print('training NN started')
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss_fn, input_image, correct_label, keep_prob, learning_rate)
        print('training NN finished')
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
