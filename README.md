## What: Temporal Autoencoder for Predicting Video

## How: Tensorflow version of CNN to LSTM to uCNN

## Why:

# Inspired by papers:

http://www.jmlr.org/proceedings/papers/v2/sutskever07a/sutskever07a.pdf
https://arxiv.org/abs/1411.4389
https://arxiv.org/abs/1504.08023
https://arxiv.org/abs/1506.04214 (like this paper with RNN but now with LSTM)
https://arxiv.org/abs/1511.06380
https://arxiv.org/abs/1511.05440
https://arxiv.org/abs/1605.08104
http://file.scirp.org/pdf/AM20100400007_46529567.pdf
https://arxiv.org/abs/1607.03597


# Uses parts of (or inspired by) the following repos:

https://github.com/tensorflow/models/blob/master/real_nvp/real_nvp_utils.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
https://github.com/machrisaa/tensorflow-vgg
https://github.com/loliverhennigh/
https://coxlab.github.io/prednet/
https://github.com/tensorflow/models/tree/master/video_prediction
https://github.com/yoonkim/lstm-char-cnn
https://github.com/anayebi/keras-extra
https://github.com/tgjeon/TensorFlow-Tutorials-for-Time-Series
https://github.com/jtoy/awesome-tensorflow
https://github.com/aymericdamien/TensorFlow-Examples

# Inspired by the following articles:

http://spectrum.ieee.org/automaton/robotics/artificial-intelligence/deep-learning-ai-listens-to-machines-for-signs-of-trouble?adbsc=social_20170124_69611636&adbid=823956941219053569&adbpl=tw&adbpr=740238495952736256

http://www.theverge.com/2016/8/4/12369494/descartes-artificial-intelligence-crop-predictions-usda

https://devblogs.nvidia.com/parallelforall/exploring-spacenet-dataset-using-digits/

# And inspired to a lesser extent the following papers:

https://arxiv.org/abs/1508.01211
https://arxiv.org/abs/1507.08750
https://arxiv.org/abs/1505.00295
www.ijcsi.org/papers/IJCSI-8-4-1-139-148.pdf
cs231n.stanford.edu/reports2016/223_Report.pdf

# Program Requirements:

* Tensorflow and related packages like python

* OpenCV

# How to run:

python main.py

And check result by making model vs. predicted video:

sh mergemov.sh

smplayer out_all.mp4
or
smplayer out_all2_fast.mp4


# Parameters:

1) In main.py:

* In main(), continuetrain: choose to use checkpoints (if exist) or not.

* Choose global flags

2) In balls.py:

* number of balls  num_balls
* SIZE: size of ball's bounding box in pixels


# Ideas and Future Work:

* Test on other models

* Try more filters

* Try more depth

* Train with geodesic acceleration (can't be done in python in tensorflow)

* Try homogenous LSTM/CNN architecture

* Include depth in CNN even if not explicitly 3D data, to avoid issues
  with overlapping pixel space causing diffusion

* Estimate velocity field in rgb, to avoid collisions most likely state as
  averaging to no motion due to L2 error's treatment of two possible
  states.

* Use entropy generation rate to train attention where can best predict.

* Try rotation, faces, and ultimately real video.




