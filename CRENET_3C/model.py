#coding=utf-8
from __future__ import print_function

import os
import time
# import random

from PIL import Image
import tensorflow.compat.v1 as tf
import numpy as np
import numpy.random as random
from utils import *
from scipy.ndimage import maximum_filter
from scipy.ndimage import minimum_filter
def concat(layers):
    return tf.concat(layers, axis=3)
def max_index_matri(im1,im2):
    one = tf.ones_like(im1)            
    zero = tf.zeros_like(im1)
    matrix= im1-concat([im2,im2,im2])
    label = tf.where(matrix <-0.0001, x=zero, y=one)     
    return label

def lrelu(x, trainbable=None):
    return tf.maximum(x*0.2,x)


# def CRENet(input_im, input_high_max, layer_num, channel=64, kernel_size=3, is_training=True):
#     # input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
#     # input_im = concat([input_max, input_im])
#     input_im = concat([input_im, input_high_max])
#     with tf.variable_scope('CRENet', reuse=tf.AUTO_REUSE):
#         conv_000 = tf.layers.conv2d(input_im, channel//2, 1, padding='same', activation=tf.nn.elu, name="first_layer_1")
#
#         conv_00 = tf.layers.conv2d(input_im, channel//2, kernel_size, padding='same', activation=tf.nn.elu, name="first_layer_2")
#
#         conv_0 = tf.layers.conv2d(input_im, channel//2, kernel_size * 3, padding='same', activation=tf.nn.elu, name="first_layer_3")
#
#         conv = concat([conv_000,conv_00,conv_0])
#
#         conv1 = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.elu, name='activated_layer_1')
#         # conv1_ba = tf.layers.batch_normalization(conv1,training=is_training)
#
#         conv2 = tf.layers.conv2d(conv1, channel*2, kernel_size,strides=2, padding='same', activation=tf.nn.elu, name='activated_layer_2')
#         conv3 = tf.layers.conv2d(conv2, channel*2, kernel_size, padding='same', activation=tf.nn.elu, name='activated_layer_3')
#
#         conv3_1 = tf.layers.conv2d(conv3, channel*4, kernel_size,strides=2,padding='same', activation=tf.nn.elu, name='activated_layer_3_1')
#         conv3_2 = tf.layers.conv2d(conv3_1, channel*4, kernel_size, padding='same', activation=tf.nn.elu, name='activated_layer_3_2')
#
#         conv3_3 = tf.layers.conv2d_transpose(conv3_2, channel*4, kernel_size,strides=2, padding='same', activation=tf.nn.elu, name='activated_layer_3_3')
#         conv3_4 = tf.layers.conv2d(conv3_3, channel*2, kernel_size, padding='same', activation=tf.nn.elu, name='activated_layer_3_4')
#
#         conv3_5 = concat([conv3_4,conv3])
#         conv4 = tf.layers.conv2d(conv3_5, channel, kernel_size,padding='same', activation=tf.nn.elu, name='activated_layer_5')
#
#         conv5 = tf.layers.conv2d_transpose(conv4, channel, kernel_size,strides=2, padding='same', activation=tf.nn.elu, name='activated_layer_4')
#         conv6_1 = tf.layers.conv2d(conv5, channel, 3,padding='same', activation=tf.nn.elu, name='activated_layer_7_1')
#
#         conv6_2=concat([conv6_1,conv1])
#
#
#         conv7_2 = tf.layers.conv2d(conv6_2, channel, 3,padding='same', activation=tf.nn.elu, name='activated_layer_7_2')
#         conv7_3 = tf.layers.conv2d(conv7_2, channel, 3,padding='same', activation=tf.nn.elu, name='activated_layer_7_3')
#
#         conv7_1=concat([conv7_3,conv])
#
#         conv7_4 = tf.layers.conv2d(conv7_1, channel, 3,padding='same', activation=tf.nn.elu, name='activated_layer_7_4')
#         conv7_5=concat([conv7_4,input_im])
#
#         conv7 = tf.layers.conv2d(conv7_5, channel, 3,padding='same', activation=tf.nn.elu, name='activated_layer_7')
#
#
#         conv8_1 = tf.layers.conv2d(conv7, channel, 3,padding='same', activation=tf.sigmoid, name='activated_layer_81')
#         conv8 = tf.layers.conv2d(conv8_1, channel, 1,padding='same', activation=tf.nn.elu, name='activated_layer_8')
#
#         conv9_1 = tf.layers.conv2d(conv7, channel, 3,padding='same', activation=tf.sigmoid, name='activated_layer_91')
#         conv9 = tf.layers.conv2d(conv9_1, channel, 1,padding='same', activation=tf.nn.elu, name='activated_layer_9')
#
#         conv10_1 = tf.layers.conv2d(conv7, channel, 3,padding='same', activation=tf.sigmoid, name='activated_layer_101')
#         conv10 = tf.layers.conv2d(conv10_1, channel, 1,padding='same', activation=tf.nn.elu, name='activated_layer_10')
#
#         convR = tf.layers.conv2d(conv8, 1, 1, padding='same', activation=None, name='recon_R_layer')
#         convG = tf.layers.conv2d(conv9, 1, 1, padding='same', activation=None, name='recon_R_layer')
#         convB = tf.layers.conv2d(conv10, 1, 1, padding='same', activation=None, name='recon_R_layer')
#
#         R_R=convR#tf.sigmoid(convR)
#         R_G=convG#tf.sigmoid(convG)
#         R_B=convB#tf.sigmoid(convB)
#
#
#         #conv9 = tf.layers.conv2d(conv7, 3, kernel_size,padding='same', activation=None, name='noise_layer_9')
#
#     R = tf.sigmoid(conv8[:,:,:,0:3])
#     L = tf.sigmoid(conv8[:,:,:,3:4])
#     #N = 0.05*tf.tanh(conv9)
#
#     return concat([R_R,R_G,R_B]), R_R#,N


def CRENet(input_im, input_high_max,layer_num, channel=64, kernel_size=3,is_training=True):
    # input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    # input_im = concat([input_max, input_im])
    input_im = concat([input_im,input_high_max])
    with tf.variable_scope('CRENet', reuse=tf.AUTO_REUSE):
        conv_0 = tf.layers.conv2d(input_im, channel/2, kernel_size, padding='same', activation=lrelu, name="first_layer")

        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")

        conv1 = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=lrelu, name='activated_layer_1')
        # conv1_ba = tf.layers.batch_normalization(conv1,training=is_training)

        conv2 = tf.layers.conv2d(conv1, channel*2, kernel_size,strides=2, padding='same', activation=lrelu, name='activated_layer_2')
        # conv2_ba = tf.layers.batch_normalization(conv2,training=is_training)

        conv3 = tf.layers.conv2d(conv2, channel*2, kernel_size, padding='same', activation=lrelu, name='activated_layer_3')
        # conv3_ba = tf.layers.batch_normalization(conv3,training=is_training)

        #conv3_1 = tf.layers.conv2d(conv3, channel*2, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_3_1')
        # conv3_ba = tf.layers.batch_normalization(conv3,training=is_training)
        #conv3_2 = tf.layers.conv2d(conv3_1, channel*2, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_3_2')
        # conv3_ba = tf.layers.batch_normalization(conv3,training=is_training)
        #conv3_3 = tf.layers.conv2d(conv3_2, channel*2, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_3_3')
        # conv3_ba = tf.layers.batch_normalization(conv3,training=is_training)

        conv4 = tf.layers.conv2d_transpose(conv3, channel, kernel_size,strides=2, padding='same', activation=lrelu, name='activated_layer_4')
        # conv4_ba = tf.layers.batch_normalization(conv4,training=is_training)
        conv4_ba2=concat([conv4,conv1])

        conv5 = tf.layers.conv2d(conv4_ba2, channel, kernel_size,padding='same', activation=lrelu, name='activated_layer_5')

        conv6=concat([conv5,conv_0])
        # conv7=concat([conv6,conv])

        conv7 = tf.layers.conv2d(conv6, channel, kernel_size,padding='same', activation=lrelu, name='activated_layer_7')

        conv8 = tf.layers.conv2d(conv7, 4, kernel_size, padding='same', activation=None, name='recon_layer')

    R = tf.sigmoid(conv8[:,:,:,0:3])
    L = tf.sigmoid(conv8[:,:,:,3:4])

    return R, L


class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.CRENet_layer_num = 5
        self.eval_ssim=0
        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        self.input_low_eq = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_eq')
        # self.input_high_eq = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_eq')
        self.input_high_max = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_nax')
        self.input_low_eq_guide = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq_guide')
        self.input_low_eq_guide_weight = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_eq_guide_weight')

        [R_low, I_low] = CRENet(self.input_low,self.input_high_max, layer_num=self.CRENet_layer_num)
        
        # vgg_inp=tf.keras.Input([None,None,3])
        # vgg= tf.keras.applications.VGG16(include_top=False,input_tensor=vgg_inp)
        # for l in vgg.layers: l.trainable=False
        # # for l in vgg.layers: print(l.name)
        # vgg_out_layer1 = vgg.get_layer(index=2).output
        # vgg_content1 = tf.keras.Model(vgg_inp,vgg_out_layer1)
        
        # vgg_out_layer2 = vgg.get_layer(index=5).output
        # vgg_content2 = tf.keras.Model(vgg_inp,vgg_out_layer2)
        
        # vgg_out_layer3 = vgg.get_layer(index=9).output
        # vgg_content3 = tf.keras.Model(vgg_inp,vgg_out_layer3)
        
        # vgg_out_layer4 = vgg.get_layer(index=13).output
        # vgg_content4 = tf.keras.Model(vgg_inp,vgg_out_layer4)
        
        # vgg_out_layer5 = vgg.get_layer(index=17).output
        # vgg_content5 = tf.keras.Model(vgg_inp,vgg_out_layer5)

        # # self.perceptual_loss1=tf.reduce_mean(tf.keras.losses.mean_squared_error(gram(vgg_content1(R_low)),gram(vgg_content1(self.input_high))))
        # # self.perceptual_loss2=tf.reduce_mean(tf.keras.losses.mean_squared_error(gram(vgg_content2(R_low)),gram(vgg_content2(self.input_high))))
        # # self.perceptual_loss3=tf.reduce_mean(tf.keras.losses.mean_squared_error(gram(vgg_content3(R_low)),gram(vgg_content3(self.input_high))))
        # # self.perceptual_loss4=tf.reduce_mean(tf.keras.losses.mean_squared_error(gram(vgg_content4(R_low)),gram(vgg_content4(self.input_high))))
        # # self.perceptual_loss5=tf.reduce_mean(tf.keras.losses.mean_squared_error(gram(vgg_content5(R_low)),gram(vgg_content5(self.input_high))))

        # self.perceptual_loss1=tf.reduce_mean(tf.keras.losses.mean_absolute_error((vgg_content1(R_low)),(vgg_content1(self.input_high))))
        # self.perceptual_loss2=tf.reduce_mean(tf.keras.losses.mean_absolute_error((vgg_content2(R_low)),(vgg_content2(self.input_high))))
        # self.perceptual_loss3=tf.reduce_mean(tf.keras.losses.mean_absolute_error((vgg_content3(R_low)),(vgg_content3(self.input_high))))
        # self.perceptual_loss4=tf.reduce_mean(tf.keras.losses.mean_absolute_error((vgg_content4(R_low)),(vgg_content4(self.input_high))))
        # self.perceptual_loss5=tf.reduce_mean(tf.keras.losses.mean_absolute_error((vgg_content5(R_low)),(vgg_content5(self.input_high))))

        # self.perceptual_loss6=tf.reduce_mean(tf.keras.losses.mean_squared_error(vgg_content2(R_low),vgg_content2(self.input_high)))



        I_low_3 = concat([I_low, I_low, I_low])

        self.output_R_low = R_low#/ratio
        self.output_I_low = I_low_3
        # self.output_N_low = N_low
        self.output_S_low_zy = (R_low * I_low_3)
        #self.output_I_delta = I_delta_3
        #self.output_S = R_low * I_delta_3

        R_low_max = tf.reduce_max(R_low, axis=3, keepdims=True)

        R_low_min = tf.reduce_min(R_low,axis=3,keepdims=True)
        # other loss
        x1 = R_low[:,:,:,0]
        y1 = R_low[:,:,:,1]
        z1 = R_low[:,:,:,2]
        L_Rlow=(x1**2+y1**2+z1**2)**0.5
        x2 = self.input_high[:,:,:,0]
        y2 = self.input_high[:,:,:,1]
        z2 = self.input_high[:,:,:,2]
        L_high=(x2**2+y2**2+z2**2)**0.5


        self.recon_loss_low = tf.reduce_mean(tf.abs((R_low - self.input_high)))#/tf.maximum(tf.layers.average_pooling2d(self.input_high,[5,5],strides=1, padding='SAME'),0.004)))

        self.recon_loss_low2 = tf.reduce_mean((1-tf.image.ssim((R_low[:,:,:,0])[:,:,:,tf.newaxis], (self.input_high[:,:,:,0])[:,:,:,tf.newaxis],max_val=1))+\
                                (1-tf.image.ssim((R_low[:,:,:,1])[:,:,:,tf.newaxis], (self.input_high[:,:,:,1])[:,:,:,tf.newaxis],max_val=1))+\
                                (1-tf.image.ssim((R_low[:,:,:,2])[:,:,:,tf.newaxis], (self.input_high[:,:,:,2])[:,:,:,tf.newaxis],max_val=1)) \
                                )#(1-tf.image.ssim(R_low, self.input_high, max_val=1))
        # other loss
        # self.recon_loss_low2 = tf.reduce_mean(tf.abs(1-tf.image.ssim(R_low, self.input_high,max_val=1)))                             
        # other loss
        # self.perceptual_loss = (self.perceptual_loss1+self.perceptual_loss2+self.perceptual_loss3+self.perceptual_loss4+self.perceptual_loss5+5*self.perceptual_loss6)
        
        self.loss_CRE_zhangyu = self.recon_loss_low + self.recon_loss_low2 #+ self.recon_loss_low3 + self.texture #+ self.recon_loss_low3# +self.texture + self.recon_loss_low3#+self.perceptual_loss#+self.recon_loss_low_min # +self.perceptual_loss+self.texture#+self.nature_loss_low_min #+self.recon_loss_low_min#+ 0.01*self.R_low_loss_smooth#+ 0.0*self.N_low_loss
        

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_CRE = [var for var in tf.trainable_variables() if 'CRENet' in var.name]
        self.train_op_CRE = optimizer.minimize(self.loss_CRE_zhangyu, var_list = self.var_CRE)

        self.sess.run(tf.global_variables_initializer())

        self.saver_CRE = tf.train.Saver(var_list = self.var_CRE)

        print("[*] Initialize model successfully...")

    def evaluate(self, epoch_num, eval_low_data,eval_high_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        ssim_junzhi=0
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)

            if train_phase == "CRE":
                pre_max = input_high_eval
                pre_min = input_low_eval
                pre = pre_min/maximum(meanFilter(pre_min)/meanFilter(pre_max+0.0001),0.004)
                # pre=tf.where(pre>0.9,0.9,pre)
                # pre=tf.where(pre<0.1,0.1,pre)
                if np.random.random()<=0.0:
                    result_1, result_2 = self.sess.run([self.output_R_low,self.recon_loss_low2], feed_dict={self.input_low: input_low_eval,self.input_high:input_high_eval,self.input_high_max:0.5*pre/pre})
                else:
                    result_1, result_2 = self.sess.run([self.output_R_low,self.recon_loss_low2], feed_dict={self.input_low: input_low_eval,self.input_high:input_high_eval,self.input_high_max:pre})

            print("SSIM-loss:%s"%result_2)
            ssim_junzhi=ssim_junzhi+result_2
            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
            # save_images(os.path.join(sample_dir, 'eval_%s_%d_%d_S.png' % (train_phase, idx + 1, epoch_num)), result_4)
            # save_images2(os.path.join(sample_dir, 'eval_%s_%d_%d_N.png' % (train_phase, idx + 1, epoch_num)), result_3)
        print('ssim_junzhi: %s '%(1-ssim_junzhi/(3*len(eval_low_data))))
        return 1-ssim_junzhi/(3*len(eval_low_data))
    def train(self, train_low_data, eval_low_data,eval_high_data,train_high_data,train_high_data_max_channel, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        # assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "CRE":
            train_op = self.train_op_CRE
            train_loss = self.loss_CRE_zhangyu
            saver = self.saver_CRE

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0
        boolflag = True
        loss = 0
        last_ssim=0
        for epoch in range(start_epoch, epoch):
            boolflag = True
            if epoch%eval_every_epoch==0 or epoch==start_epoch :
                train_high_data_max_chan = gasuss_noise2(train_high_data_max_channel)
                eval_ssim=self.evaluate(epoch + 1, eval_low_data,eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                print("self.eval_ssim:%s_%s" % (self.eval_ssim, eval_ssim))
                if eval_ssim>=self.eval_ssim:
                    self.eval_ssim=eval_ssim
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

                batch_input_high_data_max_channel = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
    
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = np.random.randint(0, h - patch_size)
                    y = np.random.randint(0, w - patch_size)

                    rand_mode = np.random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)

                    batch_input_high_data_max_channel[patch_id, :, :, :] = data_augmentation(train_high_data_max_chan[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data,train_high_data_max_channel,train_high_data_max_chan))
                        np.random.shuffle(tmp)
                        train_low_data,train_high_data,train_high_data_max_channel,train_high_data_max_chan = zip(*tmp)

                # train
                if not boolflag:
                    _ = self.sess.run([train_op], feed_dict={self.input_low: batch_input_low, \
                                                                               self.input_high: batch_input_high, \
                                                                               self.input_high_max: batch_input_high_data_max_channel, \
                                                                               self.lr: lr[epoch]})
                else:
                    boolflag=False
                    _, loss,loss1,loss2,loss3,loss4 = self.sess.run([train_op,train_loss,self.recon_loss_low,self.recon_loss_low2,self.recon_loss_low3,self.texture], feed_dict={self.input_low: batch_input_low, \
                                                                               self.input_high: batch_input_high, \
                                                                               self.input_high_max: batch_input_high_data_max_channel, \
                                                                               self.lr: lr[epoch]})

                    if math.isnan(loss):
                        break
                    print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, loss1: %.6f, loss2: %.6f, loss3: %.6f, loss4: %.6f" \
                          % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss,loss1,loss2,loss3,loss4))
                iter_num += 1


            # evalutate the model and save a checkpoint file for it
            start_step = 0
            if (epoch + 1) % eval_every_epoch == 0:
                tmp = list(zip(train_low_data, train_high_data, train_high_data_max_channel,train_high_data_max_chan))
                np.random.shuffle(tmp)
                train_low_data, train_high_data, train_high_data_max_channel,train_high_data_max_chan = zip(*tmp)

                eval_ssim=self.evaluate(epoch + 1, eval_low_data,eval_high_data, sample_dir=sample_dir, train_phase=train_phase)
                print("self.eval_ssim:%s_%s" % (self.eval_ssim, eval_ssim))
                if eval_ssim>=self.eval_ssim:
                    self.eval_ssim=eval_ssim
                    self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s%s" % (train_phase, self.eval_ssim))

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, CRE_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_CRE, _ = self.load(self.saver_CRE, './checkpoint/CRE')
        #load_model_status_Relight, _ = self.load(self.saver_Relight, './checkpoint/Relight')
        if load_model_status_CRE:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0

        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_high_test = np.expand_dims(test_high_data[idx], axis=0)
            #R_low = self.sess.run([self.output_R_low], feed_dict = {self.input_low: input_low_test})

            input_predic0 = meanFilter(np.max(input_high_test,axis=3,keepdims=True),winSize=(1,1))
            input_predic0 = meanFilter(input_high_test,winSize=(1,1))

            input_predic = input_predic0#/(input_predic0**0.7+0.00004)
            # input_predic=input_predic**0.5
            # input_predic_histeq=histeq(input_predic)
            # input_predic=np.where(input_predic<0.5,1,1)
            # input_predic=np.maximum(input_predic_histeq,input_predic)
            #ee=max(histeq(input_predic),input_predic)
            start_time = time.time()
            R_low= self.sess.run([self.output_R_low], feed_dict={self.input_low: input_low_test,self.input_high_max:input_predic})
            if(idx!=0):
                total_run_time += time.time() - start_time
            # myimage=adapthisteq(test_low_data[idx])
            if CRE_flag == CRE_flag:
                save_images(os.path.join(save_dir, name + "origna." +'png'), input_low_test)
                save_images(os.path.join(save_dir, name + "predic." +'png'), input_predic)
                save_images(os.path.join(save_dir, name + "result." +'png'), R_low)
                # save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                # save_images(os.path.join(save_dir, name + "_S_low." + suffix), output_S_low_zy,input_low_test)
                # save_images(os.path.join(save_dir, name + "adapthisteq." + suffix), myimage)
                # save_images(os.path.join(save_dir, name + "_N_low." + suffix), N_low)
                # save_images(os.path.join("./compare/after", suffix ), R_low)
                #save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                #save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
            #save_images(os.path.join(save_dir, name + "_S."   + suffix), S)
        ave_run_time = total_run_time / (float(len(test_low_data))-1)
        print("[*] Average run time: %.4f" % ave_run_time)
# def perceptual_loss(y_true,y_pred):
#     print("Note:Need to remove vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 to C:\\Users\\user_name\\.keras\\models\\")
#     vgg_inp=tf.keras.Input([None,None,3])
#     vgg= tf.keras.applications.VGG16(include_top=False,input_tensor=vgg_inp)
#     for l in vgg.layers: l.trainable=False
#     vgg_out_layer = vgg.get_layer(index=5).output
#     vgg_content = tf.keras.Model(vgg_inp,vgg_out_layer)
#     y_t=vgg_content(y_true)
#     y_p=vgg_content(y_pred)
#     loss=tf.keras.losses.mean_squared_error(y_t,y_p)
#     return tf.reduce_mean(loss)
# def gram(layer):
#     shape = tf.shape(layer)
#     num_images = shape[0]
#     width = shape[1]
#     height = shape[2]
#     num_filters = shape[3]
#     filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
#     grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

#     return grams
