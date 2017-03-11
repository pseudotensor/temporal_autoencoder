############################################################################################
# Predict video using ANNs, using tensorflow version of CNN to LSTM to uCNN
# in an autoencoder like setup
############################################################################################

# System imports
import glob
import sys
import os.path
import time
import re
import cv2
import numpy as np
import tensorflow as tf

# Local imports
import models as md
import layers as ld
import clstm

# Tensorflow FLAGS
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints',
                            """directory to store checkpoints""")
tf.app.flags.DEFINE_string('video_dir', './videos',
                            """directory to store checkpoints""")
tf.app.flags.DEFINE_integer('sizexy', 32,
                            """size x and y dimensions for model, training, and prediction""")
tf.app.flags.DEFINE_integer('sizez', 3,
                            """size z for rgb or any other such information""")
tf.app.flags.DEFINE_integer('input_seq_length', 50,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('predict_frame_start', 25,
                            """ frame number, in zero-base counting, to start using prediction as output or next input""")
tf.app.flags.DEFINE_integer('predictframes', 50,
                            """number of frames to predict""")
tf.app.flags.DEFINE_integer('max_minibatches', 1000000,
                            """maximum number of mini-batches""")
tf.app.flags.DEFINE_float('hold_prob', .8,
                            """probability for dropout""")
tf.app.flags.DEFINE_float('adamvar', .001,
                            """adamvar for dropout""")
tf.app.flags.DEFINE_integer('minibatch_size', 16,
                            """mini-batch size""")
tf.app.flags.DEFINE_integer('init_num_balls', 1,
                            """How many balls to model.""")
# Choose which model to work on
# 0 = classic bouncing balls
# 1 = rotating "ball"
tf.app.flags.DEFINE_integer('modeltype', 1,
                            """Type of model.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_integer('continuetrain', 1,
                            """Whether to continue to train (1, default) or not (0).""")


def total_parameters():
  total_parameters = 0
  for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      #print(shape)
      #print(len(shape))
      variable_parametes = 1
      for dim in shape:
          #print(dim)
          variable_parametes *= dim.value
      #print(variable_parametes)
      total_parameters += variable_parametes
  print("total_parameters=%d" % (total_parameters))
  

def tower_loss(x,x_dropout,scope):
  """Calculate the total loss on a single tower running the model.

  Args:
    scope: unique prefix string identifying the tower, e.g. 'tower0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  #######################################################
  # Create network to train
  #
  # Setup inputs
  # size of balls in x-y directions each (same)
  sizexy=FLAGS.sizexy
  # Number of rgb or depth estimation at t=0, but no convolution in this direction
  sizez=FLAGS.sizez


  cnnkernels=[3,3,3,1]
  cnnstrides=[2,1,2,1]
  cnnstrideproduct=np.product(cnnstrides)
  cnnfeatures=[8,8,8,4]
  #
  # check strides are acceptable
  testsize=sizexy
  for i in xrange(len(cnnstrides)):
    if testsize % cnnstrides[i] !=0:
      print("sizexy must be evenly divisible by each stride, in order to keep input to cnn or dcnn an integer number of pixels")
      exit
    else:
      testsize=testsize/cnnstrides[i]
  #

  dopeek=1 # whether to peek as cell state when constructing gates
  clstminput=sizexy/cnnstrideproduct # must be evenly divisible
  clstmshape=[clstminput,clstminput]
  clstmkernel=[3,3]
  clstmstride=1 # currently needs to be 1 unless implement tf.pad() or tf.nn.fractional_avg_pool()
  clstmfeatures=cnnfeatures[3] # same as features of last cnn layer fed into clstm
  #
  dcnnkernels=[1,3,3,3] # reasonably the reverse order of cnnkernels
  dcnnstrides=[1,2,1,2] # reasonably the reverse order of cnnstrides
  dcnnstrideproduct=np.product(dcnnstrides)
  # last dcnn feature is rgb again
  dcnnfeatures=[8,8,8,sizez] # reasonably the reverse order of cnnfeatures, except last cnnfeatures and last dcnnfeatures (note, features are for produced object, while kernels and strides operate on current object, hence apparent shift)
  #
  # check d-strides are acceptable
  testsize=sizexy
  for i in xrange(len(dcnnstrides)):
    if testsize % dcnnstrides[i] !=0:
      print("sizexy must be evenly divisible by each d-stride, in order to keep input to cnn or dcnn an integer number of pixels")
      exit
    else:
      testsize=testsize/dcnnstrides[i]
  #
  # ensure strides cumulate to same total product so input and output same size, because we feed output back as input
  if dcnnstrideproduct!=cnnstrideproduct:
    print("cnn and dcnn strides must match for creating input size and output same size");
    exit
  #
  #
  #




  ####################
  # Setup CLSTM
  with tf.variable_scope('clstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    # input shape, kernel filter size, number of features
    convcell = clstm.clstm(clstmshape, clstmkernel, clstmstride, clstmfeatures)
    # state: batchsize x clstmshape x clstmshape x clstmfeatures
    new_state = convcell.set_zero_state(FLAGS.minibatch_size, tf.float32) 

    # Setup deCLSTM
  with tf.variable_scope('declstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    # input shape, kernel filter size, number of features
    deconvcell = clstm.clstm(clstmshape, clstmkernel, clstmstride, clstmfeatures)
    # state: batchsize x clstmshape x clstmshape x clstmfeatures
    denew_state = deconvcell.set_zero_state(FLAGS.minibatch_size, tf.float32) 


  ########################
  # Create CNN-LSTM-dCNN for an input of input_seq_length-1 frames in n time for an output of input_seq_length-1 frames in n+1 time
  x_pred = []
  for i in xrange(FLAGS.input_seq_length-1):

    # ENCODE
    # CNN: (name, 2D square kernel filter size, stride for spatial domain, number of feature maps, name) using ELUs
    # cnn1:
    if i < FLAGS.predict_frame_start:
      # only dropout on training layers
      cnn1 = ld.cnn2d_layer(x_dropout[:,i,:,:,:], cnnkernels[0], cnnstrides[0], cnnfeatures[0], "cnn_1")
    else:
      # direct input of prior output for predictive layers
      cnn1 = ld.cnn2d_layer(x_1, cnnkernels[0], cnnstrides[0], cnnfeatures[0], "cnn_1")
    # cnn2:
    cnn2 = ld.cnn2d_layer(cnn1, cnnkernels[1], cnnstrides[1], cnnfeatures[1], "cnn_2")
    # cnn3:
    cnn3 = ld.cnn2d_layer(cnn2, cnnkernels[2], cnnstrides[2], cnnfeatures[2], "cnn_3")
    # cnn4:
    cnn4 = ld.cnn2d_layer(cnn3, cnnkernels[3], cnnstrides[3], cnnfeatures[3], "cnn_4")

    # Convolutional lstm layer (input y_0 and hidden state, output prediction y_1 and new hidden state new_state)
    y_0 = cnn4 #y_0 should be same shape as first argument in clstm.clstm() above.
    y_1, new_state = convcell(y_0, new_state, 'Conv', dopeek, 'clstm')

    # deConvolutional LSTM layer
    y_2, denew_state = deconvcell(y_1, denew_state, 'deConv', dopeek, 'declstm')

    # DECODE
    # cnn5
    cnn5 = ld.dcnn2d_layer(y_2, dcnnkernels[0], dcnnstrides[0], dcnnfeatures[0], "dcnn_4")
    # cnn6
    cnn6 = ld.dcnn2d_layer(cnn5, dcnnkernels[1], dcnnstrides[1], dcnnfeatures[1], "dcnn_3")
    # cnn7
    cnn7 = ld.dcnn2d_layer(cnn6, dcnnkernels[2], dcnnstrides[2], dcnnfeatures[2], "dcnn_2")
    # x_1 (linear act)
    x_1 = ld.dcnn2d_layer(cnn7, dcnnkernels[3], dcnnstrides[3], dcnnfeatures[3], "dcnn_1", True)
    if i >= FLAGS.predict_frame_start:
      # add predictive layer
      x_pred.append(x_1)
    # set reuse to true after first go
    if i == 0:
      tf.get_variable_scope().reuse_variables()

  # Pack-up predictive layer's results
  # e.g. for input_seq_length=10 loop 0..9, had put into x_pred i=5,6,7,8,9 (i.e. 5 frame prediction)
  x_pred = tf.stack(x_pred)
  # reshape so in order of minibatch x frame x sizex x sizey x rgb
  x_pred = tf.transpose(x_pred, [1,0,2,3,4])


  #######################################################
  # Create network to generate predicted video (TODO: could keep on only 1 gpu or on cpu)
  predictframes=FLAGS.predictframes

  ##############
  # Setup CLSTM (initialize to zero, but same convcell as in other network)
  x_pred_long = []
  new_state_pred = convcell.set_zero_state(FLAGS.minibatch_size, tf.float32)
  new_destate_pred = deconvcell.set_zero_state(FLAGS.minibatch_size, tf.float32)

  #######
  # Setup long prediction network
  for i in xrange(predictframes):

    # ENCODE
    # cnn1
    if i < FLAGS.predict_frame_start: # use known sequence for this many frames
      cnn1 = ld.cnn2d_layer(x[:,i,:,:,:], cnnkernels[0], cnnstrides[0], cnnfeatures[0], "cnn_1")
    else: # use generated sequence for rest of frames
      cnn1 = ld.cnn2d_layer(x_1_pred, cnnkernels[0], cnnstrides[0], cnnfeatures[0], "cnn_1")
    # cnn2
    cnn2 = ld.cnn2d_layer(cnn1, cnnkernels[1], cnnstrides[1], cnnfeatures[1], "cnn_2")
    # cnn3
    cnn3 = ld.cnn2d_layer(cnn2, cnnkernels[2], cnnstrides[2], cnnfeatures[2], "cnn_3")
    # cnn4
    cnn4 = ld.cnn2d_layer(cnn3, cnnkernels[3], cnnstrides[3], cnnfeatures[3], "cnn_4")

    # Convolutional lstm layer
    y_0 = cnn4
    y_1, new_state_pred = convcell(y_0, new_state_pred, 'Conv', dopeek, 'clstm')

    # deConvolutional lstm layer
    y_2, new_destate_pred = deconvcell(y_1, new_destate_pred, 'deConv', dopeek, 'declstm')

    # DECODE
    # cnn5
    cnn5 = ld.dcnn2d_layer(y_2, dcnnkernels[0], dcnnstrides[0], dcnnfeatures[0], "dcnn_4")
    # cnn6
    cnn6 = ld.dcnn2d_layer(cnn5, dcnnkernels[1], dcnnstrides[1], dcnnfeatures[1], "dcnn_3")
    # cnn7
    cnn7 = ld.dcnn2d_layer(cnn6, dcnnkernels[2], dcnnstrides[2], dcnnfeatures[2], "dcnn_2")
    # x_1_pred (linear act)
    x_1_pred = ld.dcnn2d_layer(cnn7, dcnnkernels[3], dcnnstrides[3], dcnnfeatures[3], "dcnn_1", True)
    if i >= FLAGS.predict_frame_start:
      x_pred_long.append(x_1_pred)

  # Pack-up predicted layer's results
  x_pred_long = tf.stack(x_pred_long)
  x_pred_long = tf.transpose(x_pred_long, [1,0,2,3,4])


  #######################################################
  # Setup loss Computation
  # Loss computes L2 for original sequence vs. predicted sequence over input_seq_length - (seq.start+1) frames
  # Compare x^{n+1} to xpred^n (that is supposed to be approximation to x^{n+1})
  # x: batchsize, time steps, sizexy, sizexy, sizez
  loss = tf.nn.l2_loss(x[:,FLAGS.predict_frame_start+1:,:,:,:] - x_pred[:,:,:,:,:])
  tf.summary.scalar('loss', loss)
  normalnorm=tf.nn.l2_loss(x[:,FLAGS.predict_frame_start+1:,:,:,:])
  tf.summary.scalar('normalnorm', normalnorm)
  ploss = tf.sqrt(10.0*loss/normalnorm)
  tf.summary.scalar('ploss', ploss)

  return loss,normalnorm,ploss,x_pred,x_pred_long
  


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



# Function to train autoencoder network
def autoencode(continuetrain=0,modeltype=0,init_num_balls=2):


  # Some checks
  if FLAGS.input_seq_length-1<=FLAGS.predict_frame_start:
    print("prediction frame starting point (zero starting point) beyond input size - 1, so no prediction used as next input or even used as any output to compute loss")
    exit


  # Setup graph and train
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)


    # Set training method for all towers
    opt = tf.train.AdamOptimizer(FLAGS.adamvar)
    

    # Setup independent Graph model for each gpu
    tower_grads = []
    tower_vars = []
    tower_x_pred = []
    tower_x_pred_long = []
    with tf.variable_scope(tf.get_variable_scope()): # variable scope


      # setup graph input x and x_dropout
      # x: gpus x minibatch size x input_seq_length of frames x sizex x sizey x sizez(rgb)
      x = tf.placeholder(tf.float32, [FLAGS.num_gpus, FLAGS.minibatch_size, FLAGS.input_seq_length, FLAGS.sizexy, FLAGS.sizexy, FLAGS.sizez])
    
      # Setup dropout
      hold_prob = tf.placeholder("float")
      x_dropout = tf.nn.dropout(x, FLAGS.hold_prob)

      # Go over gpus
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s%d' % ("tower", i)) as scope: # only op scope

            # Calculate the loss for one tower. This function
            # constructs the entire model but shares the variables across
            # all towers.
            with tf.variable_scope('graph'):
              towerloss,normalnorm,ploss,towerxpred,towerxpredlong = tower_loss(x[i],x_dropout[i],scope)
              tower_vars.append(towerloss)


            # Collect vars for all towers.
            tower_x_pred.append(towerxpred)
            tower_x_pred_long.append(towerxpredlong)

            # Reuse variables for the next tower (share variables across towers -- one one each gpu)
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            
            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(towerloss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            
    # Add histograms for trainable variables.
    print("trainable vars")
    for var in tf.trainable_variables():
      print(var)
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    MOVING_AVERAGE_DECAY=0.9999
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    # synchronous variable averaging
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)
      
    # List of all Variables
    variables = tf.global_variables()
    # Create saver for checkpoints and summary
    saver = tf.train.Saver(variables)

    # Save variable nstep
    nstep=0
    tf.add_to_collection('vars', nstep)

    # Summary op
    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()


    with tf.Session() as sess:
      # Initialize variables
      init = tf.global_variables_initializer()

      # Start session
      sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))


      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      
      # Initialize Network
      if continuetrain==0:
        print("Initialize network")
        sess.run(init)
      else:
        print("load network")
        # http://stackoverflow.com/questions/33759623/tensorflow-how-to-restore-a-previously-saved-model-python
        #
        # * means all if need specific format then *.csv
        list_of_files = glob.glob(FLAGS.ckpt_dir + '/model.ckpt-*.meta')
        if(len(list_of_files)==0):
          print("Initialize network")
          sess.run(init)
        else:
          latest_file = max(list_of_files, key=os.path.getctime)
          print("latest_file=%s" % (latest_file))
          #
          checkpoint_path = latest_file
          saver = tf.train.import_meta_graph(checkpoint_path)
          saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
          all_vars = tf.get_collection('vars')
          m = re.search('ckpt-([0-9]+).meta', latest_file)
          nstep = int(m.group(1))
          print("done loading network: nstep=%d" % (nstep))

      # Setup summary
      summary_writer = tf.summary.FileWriter(FLAGS.ckpt_dir, sess.graph)

      # Set number of model frames
      #modelframes=FLAGS.input_seq_length+predictframes
      modelframes=FLAGS.predictframes

      # Set how often dump video to disk
      howoftenvid=1000
      # Set how often reports error to summary
      howoftensummary=100
      # Set how often to write checkpoint file
      howoftenckpt=2000

      # count and output total number of model/graph parameters
      total_parameters()

      ###############
      # Training Loop
      startstep=nstep
      num_balls = FLAGS.init_num_balls
      for step in xrange(startstep,FLAGS.max_minibatches):
        nstep=step

        #########################
        # model-dependent code
        if step%howoftenvid==0 and step>0:
          num_balls=num_balls+1
          # limit so doesn't go beyond point where can't fit balls and reaches good_config=False always in models.py
          if num_balls>3:
            num_balls=3
          print("num_balls=%d" % (num_balls))

          
        # create input data
        tower_dat = []
        tower_datmodel = []
        with tf.variable_scope(tf.get_variable_scope()): # variable scope
          for i in xrange(FLAGS.num_gpus):
            # Generate mini-batch
            dat = md.generate_model_sample(FLAGS.minibatch_size, FLAGS.input_seq_length, FLAGS.sizexy, num_balls, FLAGS.modeltype)
  
            # Get model data for comparing to prediction if generating video
            datmodel = md.generate_model_sample(1, modelframes, FLAGS.sizexy, num_balls, FLAGS.modeltype)
            # Overwrite so consistent with ground truth for video output
            dat[0,0:FLAGS.input_seq_length] = datmodel[0,0:FLAGS.input_seq_length]

            # Collect dat for all towers.
            tower_dat.append(dat)
            tower_datmodel.append(datmodel)
            
        # pack-up input data
        tower_dat = np.asarray(tower_dat)
        tower_datmodel = np.asarray(tower_datmodel)

        
        # Train on mini-batch
        # Compute error in prediction vs. model and compute time of mini-batch task
        t = time.time()
        
        #_, lossm = sess.run(train_op,feed_dict={x:tower_dat})
        #print("sess.run on step=%d" % (step));sys.stdout.flush()
        #print("shape of tower_dat")
        #print(np.shape(tower_dat));sys.stdout.flush()
        #print("shape of x")
        #print(x.get_shape());sys.stdout.flush()
 
        _, lossm = sess.run([train_op,towerloss],feed_dict={x:tower_dat})
        elapsed = time.time() - t
        assert not np.isnan(lossm), 'Model reached lossm = NaN'


        # Store model
        if nstep%howoftensummary == 0 and nstep!=0:
          summary_str = sess.run(summary_op, feed_dict={x:tower_dat})
          summary_writer.add_summary(summary_str, nstep)
          
        # Print-out loss
        if nstep%howoftensummary == 0:
          summary_str = sess.run(summary_op, feed_dict={x:tower_dat})
          summary_writer.add_summary(summary_str, nstep) 
          print("")
          print("time per batch is " + str(elapsed) + " seconds")
          print("step=%d nstep=%d" % (step,nstep))
          print("L2 loss=%g" % (lossm))

          localnormalnorm=np.sum(tower_dat[0][0,FLAGS.predict_frame_start+1:,:,:,:]) # pull from 0th tower
          print("localnormalnorm=%d" % (localnormalnorm))
          print("L2 percent loss=%g" % (100.0*(np.sqrt(float(lossm))/float(localnormalnorm))))
        else:
          # track progress
          sys.stdout.write('.')
          sys.stdout.flush()


        # Save checkpoint
        if nstep%howoftenckpt == 0:
          print("Saving checkpoint")
          checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=nstep)  
          print("checkpoint saved to " + FLAGS.ckpt_dir)

        # Output video of model and prediction for single video in mini-batch at this step
        if nstep%howoftenvid == 0:

          # Write model video (includes given and ground truth frames)
          video_path = os.path.join(FLAGS.video_dir, '')

          #http://stackoverflow.com/questions/10605163/opencv-videowriter-under-osx-producing-no-output
          cc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
          fps=4
          sizevx=100
          sizevy=100
          sizevid=(sizevx, sizevy)

          print("")
          print("Writing model video")
          video = cv2.VideoWriter()
          success = video.open(video_path + "model_" + str(nstep) + ".mov", cc, fps, sizevid, True)
          image = tower_datmodel[0][0] # pull from 0th tower
          print(image.shape)
          for i in xrange(modelframes):
            x_1_r = np.uint8(np.minimum(1, np.maximum(image[i,:,:,:], 0)) * 255)
            new_im = cv2.resize(x_1_r, (sizevx,sizevy))
            video.write(new_im)
          video.release()

          # Write given + predicted video
          print("Writing predicted video")
          video = cv2.VideoWriter()
          success = video.open(video_path + "clstm_" + str(nstep) + ".mov", cc, fps, sizevid, True)

          # Preappend starting sequence
          image = tower_datmodel[0][0] # pull from 0th tower
          print(image.shape)
          for i in xrange(FLAGS.predict_frame_start):
            x_1_r = np.uint8(np.minimum(1,np.maximum(image[i,:,:,:], 0)) * 255)
            new_im = cv2.resize(x_1_r, (sizevx,sizevy))
            video.write(new_im)

          # Append predicted video
          image = sess.run([tower_x_pred_long],feed_dict={x:tower_dat})
          image = image[0][0][0]  # pull from 0th tower
          print(image.shape)
          for i in xrange(modelframes - FLAGS.predict_frame_start):
            x_1_r = np.uint8(np.minimum(1,np.maximum(image[i,:,:,:], 0)) * 255)
            new_im = cv2.resize(x_1_r, (sizevx,sizevy))
            video.write(new_im)
          video.release()


def main(argv=None):
  #
  continuetrain=FLAGS.continuetrain
  #
  #
  modeltype=FLAGS.modeltype
  init_num_balls=FLAGS.init_num_balls
  #
  # Setup checkpoint directory
  if tf.gfile.Exists(FLAGS.ckpt_dir):
    if continuetrain==0:
      tf.gfile.DeleteRecursively(FLAGS.ckpt_dir)
  else:
    tf.gfile.MakeDirs(FLAGS.ckpt_dir)
    continuetrain=0
    #

  # Setup video directory
  if tf.gfile.Exists(FLAGS.video_dir):
    print("Using existing video directory")
  else:
    tf.gfile.MakeDirs(FLAGS.video_dir)

  # Start training autoencoder
  autoencode(continuetrain=continuetrain,modeltype=modeltype,init_num_balls=init_num_balls)

if __name__ == '__main__':
  tf.app.run()

