import tensorflow as tf
import numpy as np
import os

#---------------------------------------------------------------------------------
# get files from training data with a shuffle
def get_files(file_dir):
    # return random shuffled image names and labels

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # random.shuffle
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

# get files from testing data
def get_files2(file_dir):
    images = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        images.append(file_dir + file)
    print("There are %d images" % (len(images)))
    
    return images

#---------------------------------------------------------------------------------

# generate batches of same size
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # transfer python.list to tensorflow
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # generate queue
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # resize image to same size
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    
    return image_batch, label_batch


#---------------------------------------------------------------------------------

import tensorflow as tf

# construct model structure
def inference(images, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
    return softmax_linear

#---------------------------------------------------------------------------------

# evaluation functions
def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def training(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

#---------------------------------------------------------------------------------

# train the model
N_CLASSES = 2
IMG_W=208
IMG_H=208
BATCH_SIZE=16
CAPACITY=2000
MAX_STEP=500  #MAX_STEP>10k
learning_rate=0.0001 #learning_rate<0.0001

def run_training():
    train_dir = 'train\\'
    logs_train_dir = 'logs_train\\'
    train, train_label = get_files(train_dir)
    train_batch, train_label_batch = get_batch(train, train_label,
                                                IMG_W, IMG_H,
                                                BATCH_SIZE, CAPACITY)
    train_logits=inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss=losses(train_logits, train_label_batch)
    train_op=training(train_loss, learning_rate)
    train_acc=evaluation(train_logits,train_label_batch)
    
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_op=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver=tf.train.Saver()

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc, tra_log = sess.run([train_op, train_loss, train_acc, train_logits])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy= %.2f%%' %(step, tra_loss, tra_acc*100))
                print(tra_log)
                summary_str=sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
                
            if step % 2000 == 0 or (step+1)==MAX_STEP:
                checkpoint_path=os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess, checkpoint_path, global_step =step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    

    
    
#---------------------------------------------------------------------------------

# evaluation part
    
from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    #Randomly pick one image from training data
    # not actually used 
    #Return: ndarray
    
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def get_one_image2(test, n):
    #pick n-th image from testing data
    #Return: ndarray
    
    img_dir = test[n]
    
    image = Image.open(img_dir)
    #plt.imshow(image)
    #plt.show()
    
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def evaluate_one_image(n):
    #Test one image against the saved models and parameters
    # you need to change the directories to yours.
    
    test_dir = 'test\\'
    test= get_files2(test_dir)
    image_array = get_one_image2(test, n)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        #you need to change the directories to yours.
        logs_train_dir = 'logs_train\\' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            print(prediction)
            return(prediction)
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f \n' %prediction[:, 0])
                
            else:
                print('This is a dog with possibility %.6f \n' %prediction[:, 1])
                
#---------------------------------------------------------------------------------
# run evaluation            
def evaluation():
    cifar10_results=[]

    for i in range(12500):  
        print('\n-----------------------')
        print(i)
        cifar10_results.append(evaluate_one_image(i))
    return(cifar10_results)