import tensorflow as tf

tf.app.flags.DEFINE_string('current_dir', '/media/euijin/Dataset/Research/server_PVS/code',
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('mode', 'train',
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('testout_path', "../test",
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_integer('current_GPU', 0,
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('train_path', '../train/model',
  """ Directory where to save the model.""")

tf.app.flags.DEFINE_string('load_model', '/V1_19.05.16',
  """ Directory where to save the model.""")

tf.app.flags.DEFINE_string('save_model', '/V1_19.05.16',
  """ Directory where to save the model.""")

tf.app.flags.DEFINE_string('summary_Name', 'summary.txt',
  """ summary text file name .""")

tf.app.flags.DEFINE_string('image_dir', '../data/set1/image',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('label_dir', '../data/set1/label',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('seg_dir', '../data/segment/set1',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('WM_path', '../data/WM/set1',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('pvs_path', '../data/pvs_seg/set1',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('image_type', 'img',
  """ image or label type ex)mhd, img, nii.""")

tf.app.flags.DEFINE_integer('npatient',1,
  """ Number of patients to train.""")

tf.app.flags.DEFINE_integer('batchsize', 4,
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_float('learningRate', 1e-4,
  """ Locations priority order in the testing dataset:""")

tf.app.flags.DEFINE_integer('decay_step',1,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_integer('epoch', 150,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_integer('fs', 3,
  """ convolution filter size.""")

tf.app.flags.DEFINE_integer('img_patchsize', 60,
  """ image patch size .""")

tf.app.flags.DEFINE_integer('train_img_pat_st',40,
  """ image patch stride.""")

tf.app.flags.DEFINE_integer('n_dbH', 6,
  """ number of dense block .""")

tf.app.flags.DEFINE_integer('n_dbA', 4,
  """  number of Whole dense block.""")

