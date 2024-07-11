# coding:utf8
import warnings

##########################.180
class DefaultConfig(object):
    #visualization parameter
    env = 'default'  # visdom environment
    vis_port =8097 # visdom port


    #load file parameter
    train_data_root = '/data2/jian/CS-MCNet/h5_2/'
    #test_data_root = '/home/jian/CS/CS-MCNet/data/new/'
    
    load_model_path = None
    pre_load_model_path = None
    
    
    save_train_root = './checkpoints/'   ####set in command
    weights = './'  ###########set in command
    
    # [yang]- for ours
    # test_data_root = '/home1/jian/UVG/'
    # save_test_root = '/home1/jian/JVCSR/CS-MC_Output/cr'
    # save_test_or_root = '/home1/jian/JVCSR/CS-MC_Output/or'
    # save_down_root = '/home1/jian/JVCSR/CS-MC_Output/down'
    
    # save_or_root_txt = '/home1/jian/JVCSR/CS-MC_Output/txt/or'
    # save_down_root_txt = '/home1/jian/JVCSR/CS-MC_Output/txt/down'
    # save_result_root_txt = '/home1/jian/JVCSR/CS-MC_Output/txt/cr'
    
    
    # [yang] - for original
    test_data_root = '/home1/jian/MCL-JCV/'
    save_test_root = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_output/cr2'
    save_test_or_root = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_originalsize/or'
    #save_down_root = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_originalsize/down'
    
 
    save_or_root_txt = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_originalsize/txt/or'
    save_result_root_txt = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_output/txt/cr2'
    #save_down_root_txt = '/home2/ok/jian/JVCSR/MCL-JCV/CS-MC_originalsize/txt/down'
    #save_whole_root = '/home1/jian/CS-MCNet/video/whole/'
    
    #training parameter
    batch_size = 4 # batch size
    num_workers = 4  # how many workers for loading data
    print_freq = 2  # print info every N batch
    max_epoch = 120
    lr = 0.01 # initial learning rate
    momentum = 0.9
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_decay_ever = 3
    weight_decay = 0  # 损失函数
    cuda_visible_devices = 2
    #test related parameter
    frame_num = 32

    #model related parameter
    cr = 0   ####set in command
    #cr = 0.015625
    height = 160
    width = 160
    blk_size = 16
    ref_size = 32
    alpha = 0.5
    noise_snr = 0

    device = 'cuda'

    #refresh config
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
    
    #save config when training
    def write_config(self,kwargs,save_root):
        f = open(save_root+"/"+"config.txt","w")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                #print(k, getattr(self, k))
                config_info = k + str(getattr(self,k))
                f.write("%s"%config_info)
                f.write("\n")
        f.close()


opt = DefaultConfig()
