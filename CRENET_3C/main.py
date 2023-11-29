#coding=utf-8
from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow.compat.v1 as tf

from model import lowlight_enhance
from utils import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="1", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='test', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=300000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=48, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=107, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/low', help='directory for testing inputs')

parser.add_argument('--predict_dir', dest='predict_dir', default='./data/pre', help='directory for testing inputs')

parser.add_argument('--CRE', dest='CRE', default=1, help='CRE flag, 0 for enhanced results only and 1 for CREposition results')

args = parser.parse_args()

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    train_low_data_eq = []
    train_low_data_clahe = []
    train_high_data_eq = []
    train_high_data_max_channel=[]

    train_low_data_eq_guide = []
    train_high_data_eq_guide = []
    train_low_data_eq_guide_weight = []
    train_low_data_eq_clahe_weight = []
    train_high_data_eq_guide_weight = []


    train_low_data_names =   glob('../../data/our485/low/*.*')+ glob('../../data/our485/high_/*.*') #+ glob('./data/our4853/low/*.*')
    #train_low_data_names.sort()
    train_high_data_names =  glob('../../data/our485/high_/*.*')+ glob('../../data/our485/high_/*.*') #+ glob('./data/our4853/low/*.*')
    #train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))


    for idx in range(len(train_low_data_names)):
        # print ("%s%s"%(train_low_data_names[idx],train_high_data_names[idx]))
        assert ((train_low_data_names[idx])[-5:-1]==(train_high_data_names[idx])[-5:-1])
        low_im = load_images2(train_low_data_names[idx])
        #low_im = white_world(low_im)
        train_low_data.append(low_im)
        high_im = load_images2(train_high_data_names[idx])
        # high_im = white_world(high_im)
        train_high_data.append(high_im)
        train_low_data_max_channel = low_im


        # Simulation of the loss of detail and noise amplification caused by the enhancement operation
        # Some unsuccessful attempts, need more work in the future

        #weight_eq_clahe=0#sigmoid(5*(mainFilter(train_low_data_max_chan,(20,20))-0.5))
        #train_low_data_max_channel = (1-weight_eq_clahe) * histeq(train_low_data_max_chan) + weight_eq_clahe * adapthisteq(train_low_data_max_chan)
        # train_low_data_max_channel = histeq(low_im[:,:,1])
        # add blur
        # if np.random.random()<=0.1:
        #     train_high_data_max_chan = meanFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=0.2:
        #     train_high_data_max_chan = mainFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=0.6:
        #     train_high_data_max_chan = mainFilter(train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(np.max(high_im,axis=2,keepdims=True)+0.0001),0.004))
        # elif np.random.random()<=0.8:
        #     train_high_data_max_chan = train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(np.max(high_im,axis=2,keepdims=True)+0.0001),0.004)
        # else:
        #     train_high_data_max_chan = np.max(high_im,axis=2,keepdims=True)
        # if np.random.random()<=0.2:
        #     train_high_data_max_chan = meanFilter(train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(np.max(high_im,axis=2,keepdims=True)+0.0001),0.004))
        # #     train_high_data_max_chan = meanFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=0.4:
        # #     train_high_data_max_chan = mainFilter(np.max(high_im,axis=2,keepdims=True))
        # # elif np.random.random()<=0.6:
        #     train_high_data_max_chan = mainFilter(train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(np.max(high_im,axis=2,keepdims=True)+0.0001),0.004))
        # elif np.random.random()<=0.8:
        #     train_high_data_max_chan = train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(np.max(high_im,axis=2,keepdims=True)+0.0001),0.004)
        # else:
        #     train_high_data_max_chan = np.max(high_im,axis=2,keepdims=True)
        if np.random.random()<=0.2:
            train_high_data_max_chan = meanFilter(high_im) # try to control contrast through mean brightness; Failed, need more complex network and bigger receptive field
        elif np.random.random()<=0.4:
            train_high_data_max_chan = mainFilter(high_im) # try to control contrast through mean brightness; Failed, need more complex network and bigger receptive field
        elif np.random.random()<=0.6:
            train_high_data_max_chan = mainFilter(train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(high_im+0.0001), 0.004)) # adding bulr, to avoid the texture loss caused by enhancemnet
        elif np.random.random()<=0.8:
            train_high_data_max_chan = train_low_data_max_channel/maximum(mainFilter(train_low_data_max_channel)/mainFilter(high_im+0.0001), 0.004) #brightness mapping, simulate noise caused by enhancement
        else:
            train_high_data_max_chan = high_im # treat reference images with gauss noise as the condition


        # if np.random.random()<=0.33:
        #     train_high_data_max_chan = meanFilter(np.max(high_im,axis=2,keepdims=True))
        # elif np.random.random()<=0.66:
        #     train_high_data_max_chan = mainFilter(np.max(high_im,axis=2,keepdims=True))
        # else:
        #     train_high_data_max_chan = np.max(high_im,axis=2,keepdims=True)
        # train_high_data_max_chan = (np.max(high_im,axis=2,keepdims=True))
        # train_low_data_eq.append(train_low_data_max_channel[:,:,:])
        train_high_data_max_channel.append(train_high_data_max_chan)

        # weight,guide = guideFilter(train_low_data_max_channel,train_low_data_max_channel,(7,7),0.1) # bigger kernal and eps, blur more 

        # train_low_data_clahe.append(adapthisteq(train_low_data_max_channel))
        # train_low_data_eq_clahe_weight.append(sigmoid(5*(mainFilter(train_low_data_max_channel,20)-0.5)))
        #print(size(weight))
        # train_low_data_eq_guide.append(guide)
        #train_high_data_eq_guide.append(histeq(np.max(low_im,axis=2,keepdims=True))) = []
        # train_low_data_eq_guide_weight.append(sigmoid(5*(weight-0.5*np.clip(weight,1.0,1.0))))
        #train_high_data_eq_guide_weight.append(histeq(np.max(low_im,axis=2,keepdims=True))) = []


    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('../../data/eval15/low/*.*')
    eval_high_data_name = glob('../../data/eval15/high/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images2(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)
        eval_high_im = load_images2(eval_high_data_name[idx])
        eval_high_data.append(eval_high_im)
    lowlight_enhance.train(train_low_data,eval_low_data,eval_high_data,train_high_data,train_high_data_max_channel, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'CRE'), eval_every_epoch=args.eval_every_epoch, train_phase="CRE")

    # lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*')
    test_low_data_name.sort()
    test_high_data_name = glob(os.path.join(args.predict_dir) + '/*')
    test_high_data_name.sort()

    test_low_data = []
    test_high_data = []

    for idx in range(len(test_low_data_name)):
        test_low_im = load_images2(test_low_data_name[idx])
        test_low_data.append(test_low_im)
        print(test_low_data_name[idx])
        test_high_im = load_images2(test_high_data_name[idx])
        test_high_data.append(test_high_im)
        print(test_high_data_name[idx])

    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir, CRE_flag=args.CRE)


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
