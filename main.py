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
tf.app.flags.DEFINE_integer('input_seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('predict_frame_start', 5,
                            """ frame number, in zero-base counting, to start using prediction as output or next input""")
tf.app.flags.DEFINE_integer('max_minibatches', 1000000,
                            """maximum number of mini-batches""")
tf.app.flags.DEFINE_float('hold_prob', .8,
                            """probability for dropout""")
tf.app.flags.DEFINE_float('adamvar', .001,
                            """adamvar for dropout""")
tf.app.flags.DEFINE_integer('minibatch_size', 16,
                            """mini-batch size""")




# Function to train autoencoder network
def autoencode(continuetrain=0,modeltype=0,num_balls=2):

  with tf.Graph().as_default():
    
    # Setup inputs
    # size of balls in x-y directions each (same)
    sizexy=FLAGS.sizexy
    # Number of rgb or depth estimation at t=0, but no convolution in this direction
    sizez=3
    # x: minibatches x input_seq_length of frames x sizex x sizey x sizez(rgb)
    x = tf.placeholder(tf.float32, [None, FLAGS.input_seq_length, sizexy, sizexy, sizez])

    # Setup dropout
    hold_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, hold_prob)

    # Some checks
    if FLAGS.input_seq_length-1<=FLAGS.predict_frame_start:
      print("prediction frame starting point (zero starting point) beyond input size - 1, so no prediction used as next input or even used as any output to compute loss")
      exit
    

    #######################################################
    # Create network to train
    #
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
        
    clstminput=sizexy/cnnstrideproduct # must be evenly divisible
    clstmshape=[clstminput,clstminput]
    clstmkernel=[3,3]
    clstmfeatures=cnnfeatures[3] # same as features of last cnn layer fed into clstm
    #
    dcnnkernels=[1,3,3,3]
    dcnnstrides=[1,2,1,2]
    dcnnstrideproduct=np.product(dcnnstrides)
    # last dcnn feature is rgb again
    dcnnfeatures=[8,8,8,sizez]
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
    x_pred = []
    with tf.variable_scope('clstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      # input shape, kernel filter size, number of features
      cell = clstm.clstm(clstmshape, clstmkernel, clstmfeatures)
      # state: batchsize x shape x shape x features
      new_state = cell.set_zero_state(FLAGS.minibatch_size, tf.float32) 

    # Create CNN-LSTM-dCNN for an input of input_seq_length-1 frames in n time for an output of input_seq_length-1 frames in n+1 time
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

      # lstm layer (input y_0 and hidden state, output prediction y_1 and new hidden state new_state)
      y_0 = cnn4 #y_0 should be same shape as first argument in clstm.clstm() above.
      y_1, new_state = cell(y_0, new_state)

      # DECODE
      # cnn5
      cnn5 = ld.dcnn2d_layer(y_1, dcnnkernels[0], dcnnstrides[0], dcnnfeatures[0], "dcnn_5")
      # cnn6
      cnn6 = ld.dcnn2d_layer(cnn5, dcnnkernels[1], dcnnstrides[1], dcnnfeatures[1], "dcnn_6")
      # cnn7
      cnn7 = ld.dcnn2d_layer(cnn6, dcnnkernels[2], dcnnstrides[2], dcnnfeatures[2], "dcnn_7")
      # x_1 (linear act)
      x_1 = ld.dcnn2d_layer(cnn7, dcnnkernels[3], dcnnstrides[3], dcnnfeatures[3], "dcnn_8", True)
      if i >= FLAGS.predict_frame_start:
        # add predictive layer
        x_pred.append(x_1)
      # set reuse to true after first go
      if i == 0:
        tf.get_variable_scope().reuse_variables()

    # Pack-up predictive layer' results
    # e.g. for input_seq_length=10 loop 0..9, had put into x_pred i=5,6,7,8,9 (i.e. 5 frame prediction)
    x_pred = tf.pack(x_pred)
    # reshape so in order of minibatch x frame x sizex x sizey x rgb
    x_pred = tf.transpose(x_pred, [1,0,2,3,4])
    

    #######################################################
    # Create network to generate predicted video
    predictframes=50

    x_pred_long = []
    new_state_pred = cell.set_zero_state(FLAGS.minibatch_size, tf.float32) 
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

      # lstm layer
      y_0 = cnn4
      y_1, new_state_pred = cell(y_0, new_state_pred)

      # DECODE
      # cnn5
      cnn5 = ld.dcnn2d_layer(y_1, dcnnkernels[0], dcnnstrides[0], dcnnfeatures[0], "dcnn_5")
      # cnn6
      cnn6 = ld.dcnn2d_layer(cnn5, dcnnkernels[1], dcnnstrides[1], dcnnfeatures[1], "dcnn_6")
      # cnn7
      cnn7 = ld.dcnn2d_layer(cnn6, dcnnkernels[2], dcnnstrides[2], dcnnfeatures[2], "dcnn_7")
      # x_1_pred (linear act)
      x_1_pred = ld.dcnn2d_layer(cnn7, dcnnkernels[3], dcnnstrides[3], dcnnfeatures[3], "dcnn_8", True)
      if i >= FLAGS.predict_frame_start:
        x_pred_long.append(x_1_pred)

    # Pack-up predicted layer's results
    x_pred_long = tf.pack(x_pred_long)
    x_pred_long = tf.transpose(x_pred_long, [1,0,2,3,4])


    #######################################################
    # Setup loss Computation
    # Loss computes L2 for original sequence vs. predicted sequence over input_seq_length - (seq.start+1) frames
    # Compare x^{n+1} to xpred^n (that is supposed to be approximation to x^{n+1})
    loss = tf.nn.l2_loss(x[:,FLAGS.predict_frame_start+1:,:,:,:] - x_pred[:,:,:,:,:])
    #tf.scalar_summary('loss', loss)
    tf.summary.scalar('loss', loss)

    # Set training method
    train_operation = tf.train.AdamOptimizer(FLAGS.adamvar).minimize(loss)
    
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
 
    # Initialize variables
    init = tf.global_variables_initializer()

    # Start session
    sess = tf.Session()

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
    modelframes=predictframes

    # Set how often dump video to disk
    howoftenvid=1000
    # Set how often reports error to summary
    howoftensummary=2000
    # Set how often to write checkpoint file
    howoftenckpt=2000

    ###############
    # Training Loop
    startstep=nstep
    for step in xrange(startstep,FLAGS.max_minibatches):
      nstep=step

      # Generate mini-batch
      dat = md.generate_model_sample(FLAGS.minibatch_size, FLAGS.input_seq_length, FLAGS.sizexy, num_balls, modeltype)
      
      # Get model data for comparing to prediction if generating video
      if nstep%howoftenvid == 0:
        datmodel = md.generate_model_sample(1, modelframes, FLAGS.sizexy, num_balls, modeltype)
        # Overwrite so consistent with ground truth for video output
        dat[0,0:FLAGS.input_seq_length] = datmodel[0,0:FLAGS.input_seq_length]
      
      # Train on mini-batch
      # Compute error in prediction vs. model and compute time of mini-batch task
      t = time.time()
      _, lossm = sess.run([train_operation, loss],feed_dict={x:dat, hold_prob:FLAGS.hold_prob})
      elapsed = time.time() - t
      assert not np.isnan(lossm), 'Model reached lossm = NaN'


      # Store model and print-out loss
      if nstep%howoftensummary == 0 and nstep != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat, hold_prob:FLAGS.hold_prob})
        summary_writer.add_summary(summary_str, nstep) 
        print("")
        print("time per batch is " + str(elapsed) + " seconds")
        print("step=%d nstep=%d" % (step,nstep))
        print("L2 loss=%g" % (lossm))

        normalnorm=np.sum(dat[0,0])
        print("normalnorm=%d" % (normalnorm))
        print("L2 percent loss=%g" % (float(lossm)/float(normalnorm)))
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
        image = datmodel[0]
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
        image = datmodel[0]
        print(image.shape)
        for i in xrange(FLAGS.predict_frame_start):
          x_1_r = np.uint8(np.minimum(1,np.maximum(image[i,:,:,:], 0)) * 255)
          new_im = cv2.resize(x_1_r, (sizevx,sizevy))
          video.write(new_im)

        # Append predicted video
        dat_gif = dat
        image = sess.run([x_pred_long],feed_dict={x:dat_gif, hold_prob:FLAGS.hold_prob})
        image = image[0][0]
        print(image.shape)
        for i in xrange(modelframes - FLAGS.predict_frame_start):
          x_1_r = np.uint8(np.minimum(1,np.maximum(image[i,:,:,:], 0)) * 255)
          new_im = cv2.resize(x_1_r, (sizevx,sizevy))
          video.write(new_im)
        video.release()


def main(argv=None):
  #
  # Choose to continue training (1) or not (0)
  continuetrain=1
  #
  #
  # Choose which model to work on
  # 0 = classic bouncing balls
  # 1 = rotating "ball"
  modeltype=1
  # Number of balls
  num_balls=1
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
  autoencode(continuetrain=continuetrain,modeltype=modeltype,num_balls=num_balls)

if __name__ == '__main__':
  tf.app.run()


