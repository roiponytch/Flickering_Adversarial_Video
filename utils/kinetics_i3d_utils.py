import cv2
import numpy as np
import imageio
import tensorflow as tf
import i3d
import yaml
from easydict import EasyDict as edict

_IMAGE_SIZE = 224
_BATCH_SIZE = 1

_SAMPLE_VIDEO_FRAMES =90 #79 # 79 #90 #90 #250 #90 #79
_BASE_PATCH_FRAMES = _SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES #_SAMPLE_VIDEO_FRAMES # 1# _SAMPLE_VIDEO_FRAMES # 1:for sticker _SAMPLE_VIDEO_FRAMES # 1
_IND_START = 0  # 0 #50
_IND_END =_SAMPLE_VIDEO_FRAMES

_LABEL_MAP_PATH = 'data/label_map.txt'

NUM_CLASSES = 400


def load_config(yml_path):
    with open(yml_path,'r') as f:
        cfg = edict(yaml.load(f))
        
    return cfg

def load_i3d_model(num_classes,eval_type='rgb', scope='RGB',spatial_squeeze=True, final_endpoint='Logits'):
    with tf.variable_scope(scope):
        i3d_model = i3d.InceptionI3d(
          num_classes, spatial_squeeze=spatial_squeeze, final_endpoint=final_endpoint)

    dummy_input = tf.placeholder(
        tf.float32,
        shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    i3d_model(dummy_input, is_training=False, dropout_keep_prob=1.0)


    return i3d_model

def init_model(model,sess, ckpt_path, eval_type='rgb'):
    rgb_variable_map = {}

    for variable in model.get_all_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    for variable in model.graph.get_collection_ref('moving_average_variables'):

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    saver.restore(sess,ckpt_path)


tf_config = tf.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99

def load_kinetics_classes(eval_type='rgb'):
    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    return kinetics_classes

class kinetics_i3d():
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt', batch_size=1
               ):
    """Initializes Unit3D module."""
    super(kinetics_i3d, self).__init__()
    self.ckpt_path=ckpt_path
    _BATCH_SIZE = batch_size
    scope ='RGB'
    self.sess = tf.Session(config  = tf_config)
    with tf.variable_scope(scope):

        default_adv_flag = tf.constant(1.0,dtype=tf.float32)
        self.adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)
    
        # RGB input has 3 channels.
        self.rgb_input = tf.placeholder(tf.float32,
            shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
        self.eps_rgb = tf.Variable(tf.zeros(shape=[_BASE_PATCH_FRAMES, 1, 1, 3], dtype=tf.float32),name='eps')
        # extend_eps_rgb = tf.tile(eps_rgb, [9, 1, 1, 1])
        self.eps_rgb_clip = tf.clip_by_value(self.eps_rgb,clip_value_min = -0.5,
                                              clip_value_max = 0.5)
    
        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    
        indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
        random_shift = tf.random_uniform(dtype=tf.int32, minval=0, maxval=self.rgb_input.shape[1].value, shape=[])
        self.cyclic_rgb_input = tf.roll(self.rgb_input,shift=random_shift,axis=1)
    
        cyclic_flag_default = tf.constant(0.0, dtype=tf.float32)
        self.cyclic_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_flag',
                                                   shape=cyclic_flag_default.shape)
    
        self.adversarial_inputs_rgb = tf.clip_by_value(self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input + self.adv_flag * (mask_rgb * self.eps_rgb_clip),
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')

    
    self.rgb_model = load_i3d_model(num_classes=400)
    init_model(model=self.rgb_model,sess=self.sess, ckpt_path=self.ckpt_path,eval_type='rgb')
    
    self.model_logits, _ = self.rgb_model(self.adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    self.softmax = tf.nn.softmax(logits = self.model_logits)
    self.sess.run(self.eps_rgb.initializer)

    self.kinetics_classes =load_kinetics_classes()
    self.labels = tf.placeholder(tf.int64, (_BATCH_SIZE,), name='labels')
    
    
    
    self.one_hot_label = tf.one_hot(self.labels,NUM_CLASSES)
    self.label_prob =tf.boolean_mask(self.softmax,tf.cast(self.one_hot_label,dtype=tf.bool))
    self.label_logits = tf.boolean_mask(self.model_logits,tf.cast(self.one_hot_label,dtype=tf.bool))
    
    self.max_non_label_prob = tf.reduce_max(self.softmax-self.one_hot_label,axis=-1)
    self.max_non_label_logits = tf.reduce_max(self.model_logits-self.one_hot_label,axis=-1)


    self.perturbation = self.eps_rgb

    
    ######### Regularization ###########
    #thickness - loss term
    self.norm_reg = tf.reduce_mean((self.perturbation)**2)+1e-12
    
    self.perturbation_roll_right =  tf.roll(self.perturbation,1,axis=0)
    self.perturbation_roll_left =  tf.roll(self.perturbation,-1,axis=0)
    
    #1st order diff - loss term
    self.diff_norm_reg = tf.reduce_mean((self.perturbation -self.perturbation_roll_right)**2)+1e-12
    
    #2nd order diff - loss term
    self.laplacian_norm_reg =tf.reduce_mean( (-2*self.perturbation +self.perturbation_roll_right+self.perturbation_roll_left)**2)+1e-12
    
    
    ########## Metric ####################
    self.roughness = tf.reduce_mean(tf.abs(self.perturbation - self.perturbation_roll_right))
    self.thickness = tf.reduce_mean(tf.abs(self.perturbation))


    
  def __call__(self, inputs ,adv_flag=0):
         self.prob = self.sess.run(self.softmax,feed_dict={self.rgb_input:inputs, self.adv_flag:adv_flag})
         return self.prob
    
  def get_kinetics_classes(self):
        return self.kinetics_classes
    
  def evaluate(self,next_element_val, targeted_attack=False, target_class_id=None,cyclic=0, exclude_misclassify=True):
    
      try:
          
          miss=0
          total_val_vid =0
          
          while True:
        
            rgb_sample, sample_label = self.sess.run(next_element_val)

            feed_dict_for_adv_eval = {self.rgb_input: rgb_sample, self.adv_flag:1, self.cyclic_flag:cyclic}
            prob = self.sess.run(feed_dict=feed_dict_for_adv_eval, fetches=self.softmax)
            
            if targeted_attack:
                miss_cond = prob.argmax(axis=-1)==target_class_id
            else:
                miss_cond = prob.argmax(axis=-1)!=sample_label
            
            if exclude_misclassify:
                prob_clean = self.sess.run(feed_dict= {self.rgb_input: rgb_sample, self.adv_flag: 0, self.cyclic_flag:0},
                                       fetches=self.softmax)            
                valid_videos = prob_clean.argmax(axis=-1)==sample_label                
                miss+=(np.logical_and(miss_cond,valid_videos)).sum()
                total_val_vid+=valid_videos.sum()
            else:
                miss+=miss_cond.sum()
                total_val_vid+=miss_cond.shape[0]

      except tf.errors.OutOfRangeError:
          miss_rate = miss/total_val_vid
          pass
      
      return miss_rate, total_val_vid

 
  def improve_adversarial_loss(self, margin=0.05,targeted=False, logits=False):
    
    if targeted:
          if logits:
              self.to_min_elem = self.max_non_label_logits
              self.to_max_elem = self.label_logits
              loss_margin= tf.log(1.+ margin*(1./self.label_prob))
          else:
              self.to_min_elem = self.max_non_label_prob
              self.to_max_elem = self.label_prob
              loss_margin = margin
         
          self.to_min_prob = self.max_non_label_prob
          self.to_max_prob = self.label_prob
    else:
          if logits:
              self.to_min_elem = self.label_logits
              self.to_max_elem = self.max_non_label_logits
              loss_margin= tf.log(1.+ margin*(1./self.max_non_label_prob))
          else:
              self.to_min_elem = self.label_prob
              self.to_max_elem = self.max_non_label_prob
              loss_margin = margin
          
          self.to_min_prob = self.label_prob
          self.to_max_prob = self.max_non_label_prob
          
    self.l_1= 0.0
    self.l_2 = (( self.to_min_elem-( self.to_max_elem-loss_margin))**2)/loss_margin
    self.l_3 = self.to_min_elem -( self.to_max_elem-loss_margin)
    
    self.adversarial_loss = tf.maximum(self.l_1, tf.minimum(self.l_2,self.l_3))
    self.adversarial_loss_total = tf.reduce_sum(self.adversarial_loss)
    
    
    return self.adversarial_loss_total

  def ce_adversarial_loss(self,targeted=False): 
    if targeted:
        self.ce_adversarial_loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
        self.to_min_prob = self.max_non_label_prob
        self.to_max_prob = self.label_prob
    else:
        self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
        self.to_min_prob = self.label_prob
        self.to_max_prob = self.max_non_label_prob
        
    self.adversarial_loss_total = tf.reduce_sum(self.ce_adversarial_loss)
    
    return self.adversarial_loss_total
  
          
    
class kinetics_i3d_inference():
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt',
               ):
    """Initializes Unit3D module."""
    super(kinetics_i3d_inference, self).__init__()
    self.ckpt_path=ckpt_path
    scope ='RGB'
    self.sess = tf.Session(config  = tf_config)
    with tf.variable_scope(scope):

        default_adv_flag = tf.constant(1.0,dtype=tf.float32)
        self.adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)
    
        # RGB input has 3 channels.
        self.rgb_input = tf.placeholder(tf.float32,
            shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
        self.eps_rgb = tf.Variable(tf.zeros(shape=[_BASE_PATCH_FRAMES, 1, 1, 3], dtype=tf.float32),name='eps')
        # extend_eps_rgb = tf.tile(eps_rgb, [9, 1, 1, 1])
        
    
        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    
        indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
        random_shift = tf.random_uniform(dtype=tf.int32, minval=0, maxval=self.rgb_input.shape[1].value, shape=[])
        self.cyclic_rgb_input = tf.roll(self.rgb_input,shift=random_shift,axis=1)
    
        cyclic_flag_default = tf.constant(0.0, dtype=tf.float32)
        self.cyclic_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_flag',
                                                   shape=cyclic_flag_default.shape)
    
        self.adversarial_inputs_rgb = tf.clip_by_value(self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input + self.adv_flag * (mask_rgb * self.eps_rgb),
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')

    
    self.rgb_model = load_i3d_model(num_classes=400)
    init_model(model=self.rgb_model,sess=self.sess, ckpt_path=self.ckpt_path,eval_type='rgb')
    
    self.model_logits, _ = self.rgb_model(self.adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    self.softmax = tf.nn.softmax(logits = self.model_logits)
    self.sess.run(self.eps_rgb.initializer)

    self.kinetics_classes =load_kinetics_classes()

    
  def __call__(self, inputs ,adv_flag=0):
         self.prob = self.sess.run(self.softmax,feed_dict={self.rgb_input:inputs, self.adv_flag:adv_flag})
         return self.prob
    
  def get_kinetics_classes(self):
        return self.kinetics_classes
    
    