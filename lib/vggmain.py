#DATA:
    #1. cifar10(binary version):https://www.cs.toronto.edu/~kriz/cifar.html
    #2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM

# TO Train and test:
    #0. get data ready, get paths ready !!!
    #1. run training_and_val.py and call train() in the console
    #2. call evaluate() in the console to test

#%%

import os
import os.path
import numpy as np
import tensorflow as tf
import input_train_val_split
import tools
import VGG
import math

#%%
N_CLASSES = 2
IMG_W =  208 # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
RATIO = 0.2 # take 20% of dataset as validation data
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.001 # with current parameters, it is suggested to use learning rate<0.0001
IS_PRETRAIN = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]
#%%
pre_trained_weights = './vgg16_pretrain/vgg16.npy'
train_dir = './data/train/'
train_log_dir = './logs/train/'
val_log_dir = './logs/val/'
test_dir = './data/test1/'


def train():
    with tf.name_scope('input'):
        train, train_label, val, val_label = input_train_val_split.get_files(train_dir, RATIO)
        tra_image_batch, tra_label_batch = input_train_val_split.get_batch(
                                                                          train,
                                                                          train_label,
                                                                          IMG_W,
                                                                          IMG_H,
                                                                          BATCH_SIZE,
                                                                          CAPACITY)
        val_image_batch, val_label_batch = input_train_val_split.get_batch(
                                                                          val,
                                                                          val_label,
                                                                          IMG_W,
                                                                          IMG_H,
                                                                          BATCH_SIZE,
                                                                          CAPACITY)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE,N_CLASSES])

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, my_global_step)


    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tools.load_with_skip(pre_trained_weights, sess, ['fc8'])
        print("load weights done")


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)


        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x:tra_images, y_:tra_labels})
                if step % 2 == 0 or (step + 1) == MAX_STEP:

                    print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    _, summary_str = sess.run([train_op, summary_op], feed_dict={x: tra_images, y_: tra_labels})
                    tra_summary_writer.add_summary(summary_str, step)

                if step % 4 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x:val_images,y_:val_labels})

                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                    _, summary_str = sess.run([train_op, summary_op], feed_dict={x: val_images, y_: val_labels})
                    val_summary_writer.add_summary(summary_str, step)

                if step % 8 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            coord.request_stop()

        coord.join(threads)

#%%


#%%   Test the accuracy on test dataset. got about 85.69% accuracy.

def evaluate():
    with tf.Graph().as_default():
        log_dir = './logs/train/'
        test_dir = './data/test1/'
        n_test = 12500
        prob_list = []
        test = input_train_val_split.get_test_files(test_dir)
        test_image_batch = input_train_val_split.get_test_batch(test,
                                                                IMG_W,
                                                                IMG_H,
                                                                TEST_BATCH_SIZE,
                                                                capacity = CAPACITY)

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        logits = VGG.VGG16N(test_image_batch, N_CLASSES, IS_PRETRAIN)
        logits = tf.nn.softmax(logits)

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(train_log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            try:
                print('\nEvaluating......')
                num_step = int(math.floor(n_test / TEST_BATCH_SIZE))
                num_sample = num_step*BATCH_SIZE
                step = 0
                while step < num_step and not coord.should_stop():
                    test_images = sess.run(test_image_batch)
                    prob = sess.run(logits, feed_dict={x:test_images})
                    prob_list.extend(prob)
                    step += 1
                print('Total testing samples: %d' %num_sample)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':

    evaluate()
