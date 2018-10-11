import tensorflow as tf


#"/media/euijin/Local Disk/DGIST/Research/KneeMR/kneeMR_SKI-20170814T043101Z-001/kneeMR_SKI" #euijin linux
#"/user/home2/euijin/Research/server_PVS/code" #server

tf.app.flags.DEFINE_string('testout_path', "../test_out/enhancement",
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('val_img_path', "../data/set2/image",
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('val_label_path', "../data/set2/label",
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('val_seg_path', '../data/segment/set2',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('val_pvs_path', '../data/pvs_seg/set2',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('val_WM_path', '../data/WM/set2',
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_string('mode', 'train',
  """ Directory where to find the model.""")

tf.app.flags.DEFINE_string('current_dir', 'D:/DGIST/Research/server_PVS/code',
  """ Directory where to find the model.""")


tf.app.flags.DEFINE_string('network_num', "2",
  """ 0:SRdense 3d, 1:SRdense 2d, 2: SRCNN.""")


tf.app.flags.DEFINE_integer('current_GPU', 0,
  """ Directory where to find the model.""")

#D:/DGIST/Research/server_PVS
tf.app.flags.DEFINE_string('model_dir', '../train_out/model/SRCNN_test',
  """ Directory where to save the model.""")

tf.app.flags.DEFINE_string('save_dir', '../train_out/model/SRCNN_test',
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

tf.app.flags.DEFINE_integer('npatient', 1,
  """ Number of patients to train.""")

tf.app.flags.DEFINE_integer('n_patient_val', 8,
  """ Number of patients to validate.""")

tf.app.flags.DEFINE_integer('batchsize', 20,
  """ Directory where to find the dataset.""")

tf.app.flags.DEFINE_float('learningRate', 1e-4,
  """ Locations priority order in the testing dataset:""")

tf.app.flags.DEFINE_integer('decay_step', 1,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_float('factor', 0.002,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_integer('epoch', 20,
  """ Number of steps to update the learning rate.""")

tf.app.flags.DEFINE_integer('fs', 3,
  """ convolution filter size.""")

tf.app.flags.DEFINE_integer('img_patchsize', 60,
  """ image patch size .""")

tf.app.flags.DEFINE_integer('train_img_pat_st',70,
  """ image patch stride.""")

tf.app.flags.DEFINE_integer('n_dbH', 6,
  """ number of dense block .""")

tf.app.flags.DEFINE_integer('n_dbA', 1,
  """  number of Whole dense block.""")



