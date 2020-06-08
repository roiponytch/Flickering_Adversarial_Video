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

  def __init__(self, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt', batch_size=1, init_model=True,rgb_input=None, labels=None
               ,cyclic_flag_default_c=0.0,cyclic_pert_flag_default_c=0.0,default_adv_flag_c=1.0 ):
    """Initializes Unit3D module."""
    super(kinetics_i3d, self).__init__()
    self.ckpt_path=ckpt_path
    _BATCH_SIZE = batch_size
    scope ='RGB'
    self.sess = tf.Session(config  = tf_config)
    with tf.variable_scope(scope):


        default_adv_flag = tf.constant(default_adv_flag_c,dtype=tf.float32)
        self.adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)
    
        # RGB input has 3 channels.
        if rgb_input != None:
            self.rgb_input = rgb_input
        else:
            self.rgb_input = tf.placeholder(tf.float32,
                shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
        self.eps_rgb = tf.get_variable(initializer=tf.zeros(shape=[_BASE_PATCH_FRAMES, 1, 1, 3], dtype=tf.float32), name='eps')
        # self.eps_rgb = tf.get_variable(initializer=tf.zeros(shape=[_BASE_PATCH_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32), name='eps')

        # extend_eps_rgb = tf.tile(eps_rgb, [9, 1, 1, 1])
        self.eps_rgb_clip = tf.clip_by_value(self.eps_rgb,clip_value_min = -0.4,
                                              clip_value_max = 0.4)
    
        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    
        indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
        random_shift = tf.random_uniform(dtype=tf.int32, minval=0, maxval=_SAMPLE_VIDEO_FRAMES, shape=[])
        self.cyclic_rgb_input = tf.roll(self.rgb_input,shift=random_shift,axis=1)
    
        cyclic_flag_default = tf.constant(cyclic_flag_default_c, dtype=tf.float32)
        self.cyclic_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_flag',
                                                   shape=cyclic_flag_default.shape)
        
        self.input_pert = mask_rgb * self.eps_rgb_clip
        
        # self.adversarial_inputs_rgb = tf.clip_by_value(self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input + self.adv_flag * (self.input_pert),
        #                                           clip_value_min = -1.0,
        #                                           clip_value_max = 1.0,
        #                                           name='adversarial_input')
        
        
        self.random_shift_2 = tf.random_uniform(dtype=tf.int32, minval=0, maxval=self.input_pert.shape[0].value, shape=[])
        self.cyclic_input_pert = tf.roll(self.input_pert,shift=self.random_shift_2,axis=0)
        self.cyclic_pert_flag = tf.placeholder_with_default(cyclic_pert_flag_default_c, name='cyclic_pert_flag',
                                                   shape=cyclic_flag_default.shape)
        
        self.clean_input = self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input
        
        self.input_pert_maybe_cyclic = self.cyclic_pert_flag*self.cyclic_input_pert+ (1-self.cyclic_pert_flag)*(self.input_pert)
                 
        self.adversarial_inputs_rgb = tf.clip_by_value( self.clean_input + self.adv_flag *self.input_pert_maybe_cyclic ,
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')
        
        
        

    self.net_input = tf.convert_to_tensor(self.adversarial_inputs_rgb)
    self.rgb_model = load_i3d_model(num_classes=400)

    self.model_logits,self.end_points = self.rgb_model(self.adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    self.model_logits_clean, _ = self.rgb_model(self.rgb_input, is_training=False, dropout_keep_prob=1.0)
    self.softmax = tf.nn.softmax(logits = self.model_logits)
    self.softmax_clean = tf.nn.softmax(logits=self.model_logits_clean)

    self.kinetics_classes =load_kinetics_classes()

    if labels == None:
        self.labels = tf.placeholder(tf.int64, (_BATCH_SIZE,), name='labels')
    else:
        self.labels = labels
    
    
    
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
    
    
    # L1,2
    self.loss_L12 = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(self.input_pert), axis=[1, 2, 3])))

    
    ########## Metric ####################
    self.roughness = tf.reduce_mean(tf.abs(self.perturbation - self.perturbation_roll_right))
    self.thickness = tf.reduce_mean(tf.abs(self.perturbation))


    
    self.roughness_relative = (self.roughness/2.0)*100
    self.thickness_relative = (self.thickness/2.0)*100

    if init_model:
        self.sess.run(self.eps_rgb.initializer)
        self.init_model()


  def init_model(self):
      init_model(model=self.rgb_model, sess=self.sess, ckpt_path=self.ckpt_path, eval_type='rgb')

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
              loss_margin= tf.log(1.+ margin*(1./(0.00001+self.max_non_label_prob)))
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
        # self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
        # self.to_min_prob = self.label_prob
        # self.to_max_prob = self.max_non_label_prob
        
        self.ce_adversarial_loss = -tf.log(1-self.label_prob + 1e-6) 
        # self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
        self.to_min_prob = self.label_prob
        self.to_max_prob = self.max_non_label_prob
        
    self.adversarial_loss_total = tf.reduce_mean(self.ce_adversarial_loss)
    
    return self.adversarial_loss_total
class kinetics_i3d_L12():
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt', batch_size=1, init_model=True,rgb_input=None, labels=None
               ,cyclic_flag_default_c=0.0,default_adv_flag_c=1.0 ):
    """Initializes Unit3D module."""
    super(kinetics_i3d_L12, self).__init__()
    self.ckpt_path=ckpt_path
    _BATCH_SIZE = batch_size
    scope ='RGB'
    self.sess = tf.Session(config  = tf_config)
    with tf.variable_scope(scope):


        default_adv_flag = tf.constant(default_adv_flag_c,dtype=tf.float32)
        self.adv_flag = tf.placeholder_with_default(default_adv_flag,shape=default_adv_flag.shape)
    
        # RGB input has 3 channels.
        if rgb_input != None:
            self.rgb_input = rgb_input
        else:
            self.rgb_input = tf.placeholder(tf.float32,
                shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
        # self.eps_rgb = tf.get_variable(initializer=tf.zeros(shape=[_BASE_PATCH_FRAMES, 1, 1, 3], dtype=tf.float32), name='eps')
        self.eps_rgb = tf.get_variable(initializer=tf.constant(1e-8,shape=[_BASE_PATCH_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3], dtype=tf.float32), name='eps')

        # extend_eps_rgb = tf.tile(eps_rgb, [9, 1, 1, 1])
        self.eps_rgb_clip = self.eps_rgb
        # self.eps_rgb_clip = tf.clip_by_value(self.eps_rgb,clip_value_min = -0.4,
        #                                       clip_value_max = 0.4)
    
        mask = tf.ones(shape=[_SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3])
    
        indices = np.linspace(_IND_START,_IND_END,_IND_END-_IND_START+1)
        mask_indecator = tf.one_hot(indices =indices, depth=_SAMPLE_VIDEO_FRAMES)
        mask_indecator = tf.reduce_sum(mask_indecator, reduction_indices=0)
        mask_indecator = tf.reshape(mask_indecator, [_SAMPLE_VIDEO_FRAMES,1,1,1])
        mask_rgb = tf.convert_to_tensor(mask*mask_indecator,name='eps_mask') # same shape as input
        # adversarial_inputs_rgb = tf.nn.tanh(rgb_input + adv_flag * (mask_rgb * eps_rgb),name='adversarial_input')
        random_shift = tf.random_uniform(dtype=tf.int32, minval=0, maxval=_SAMPLE_VIDEO_FRAMES, shape=[])
        self.cyclic_rgb_input = tf.roll(self.rgb_input,shift=random_shift,axis=1)
    
        cyclic_flag_default = tf.constant(cyclic_flag_default_c, dtype=tf.float32)
        self.cyclic_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_flag',
                                                   shape=cyclic_flag_default.shape)
        
        self.input_pert = mask_rgb * self.eps_rgb_clip
        self.adversarial_inputs_rgb = tf.clip_by_value(self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input + self.adv_flag * (self.input_pert),
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')
        
        


        

    self.net_input = tf.convert_to_tensor(self.adversarial_inputs_rgb)
    self.rgb_model = load_i3d_model(num_classes=400)

    self.model_logits,self.end_points = self.rgb_model(self.adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    self.model_logits_clean, _ = self.rgb_model(self.rgb_input, is_training=False, dropout_keep_prob=1.0)
    self.softmax = tf.nn.softmax(logits = self.model_logits)
    self.softmax_clean = tf.nn.softmax(logits=self.model_logits_clean)

    self.kinetics_classes =load_kinetics_classes()

    if labels == None:
        self.labels = tf.placeholder(tf.int64, (_BATCH_SIZE,), name='labels')
    else:
        self.labels = labels
    
    
    
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
    
    
    # L1,2
    self.loss_L12 = tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(self.perturbation), axis=[1, 2, 3])))+1e-12

    
    ########## Metric ####################
    self.roughness = tf.reduce_mean(tf.abs(self.perturbation - self.perturbation_roll_right))
    self.thickness = tf.reduce_mean(tf.abs(self.perturbation))
    
    self.roughness_relative = (self.roughness/2.0)*100
    self.thickness_relative = (self.thickness/2.0)*100

    if init_model:
        self.sess.run(self.eps_rgb.initializer)
        self.init_model()


  def init_model(self):
      init_model(model=self.rgb_model, sess=self.sess, ckpt_path=self.ckpt_path, eval_type='rgb')

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
              loss_margin= tf.log(1.+ margin*(1./(0.00001+self.max_non_label_prob)))
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
        
        self.ce_adversarial_loss = -tf.log(1-self.label_prob + 1e-6) 
        # self.ce_adversarial_loss = -1.0*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.model_logits)
        self.to_min_prob = self.label_prob
        self.to_max_prob = self.max_non_label_prob
        
    self.adversarial_loss_total = tf.reduce_mean(self.ce_adversarial_loss)
    
    return self.adversarial_loss_total


def miss_cls_fn(predictions=None, labels=None, weights=None, targeted= False):
    one_const  = tf.constant(1, dtype=tf.float32)
    if targeted:
        acc, update_op =tf.metrics.accuracy(predictions=predictions, labels=labels, weights=weights)
        return (acc, update_op)
    
    else:
        acc, update_op =tf.metrics.accuracy(predictions=predictions, labels=labels, weights=weights)

        return (acc, update_op)


from tensorflow.python.training import distribution_strategy_context

def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across towers."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution._outer_control_flow_context is None:
        with tf.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      tf.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_tower_context().merge_call(fn, *args)


          
    
class kinetics_i3d_inference():
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""

  def __init__(self, ckpt_path='data/checkpoints/rgb_imagenet/model.ckpt', batch_size=_BATCH_SIZE
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
            shape=(batch_size, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    
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
        
        
        self.random_shift_2 = tf.random_uniform(dtype=tf.int32, minval=0, maxval=self.eps_rgb.shape[0].value, shape=[])
        self.cyclic_eps_rgb = tf.roll(self.eps_rgb,shift=self.random_shift_2,axis=0)
        self.cyclic_eps_flag = tf.placeholder_with_default(cyclic_flag_default, name='cyclic_eps_flag',
                                                   shape=cyclic_flag_default.shape)
        
        self.clean_input = self.cyclic_flag*self.cyclic_rgb_input +(1-self.cyclic_flag)*self.rgb_input
        
        self.epsilon = self.cyclic_eps_flag*(mask_rgb * self.cyclic_eps_rgb)+ (1-self.cyclic_eps_flag)*(mask_rgb * self.eps_rgb)
                 
        self.adversarial_inputs_rgb = tf.clip_by_value( self.clean_input + self.adv_flag *self.epsilon ,
                                                  clip_value_min = -1.0,
                                                  clip_value_max = 1.0,
                                                  name='adversarial_input')

    
    self.rgb_model = load_i3d_model(num_classes=400)
    
    
    self.model_logits, _ = self.rgb_model(self.adversarial_inputs_rgb, is_training=False, dropout_keep_prob=1.0)
    self.softmax = tf.nn.softmax(logits = self.model_logits)
    self.sess.run(self.eps_rgb.initializer)

    self.kinetics_classes =load_kinetics_classes()

    init_model(model=self.rgb_model,sess=self.sess, ckpt_path=self.ckpt_path,eval_type='rgb')
    
  def __call__(self, inputs ,adv_flag=0, cyclic_input_flag=0, cyclic_eps_flag=0):
         self.prob = self.sess.run(self.softmax,feed_dict={self.rgb_input:inputs,
                                                           self.adv_flag:adv_flag,
                                                           self.cyclic_flag:cyclic_input_flag,
                                                           self.cyclic_eps_flag:cyclic_eps_flag})
         return self.prob
    
  def get_kinetics_classes(self):
        return self.kinetics_classes
    
    