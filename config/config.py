import numpy as np
from easydict import EasyDict


__C = EasyDict()

config = __C

__C.DEBUG = False

__C.VERBOSE = True

__C.PRETRAINED_RPN = True

__C.TRAIN = EasyDict()

__C.TEST = EasyDict()

__C.USE_GPU_NMS = True

### DATA SET ####
__C.TRAIN.USE_FLIPPED = True

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

__C.TRAIN.USE_ALL_GT = True

### REGULARIZATION AND OPTIMATION ####

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

__C.TRAIN.DECAY_LRS = [6,]

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False


##### RPN OPTION ####
__C.ANCHOR_SCALES = np.array([8, 16, 32])

__C.ANCHOR_RATIOS = [0.5, 1, 2]

# Down-sampling ratio
__C.FEAT_STRIDE = [16, ]

__C.TRAIN.RPN_CLOBBER_POSITIVES = False

__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

__C.TRAIN.RPN_BATCHSIZE = 256

__C.TRAIN.RPN_FG_FRACTION = 0.5

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

__C.TRAIN.RPN_NMS_THRESH = 0.7

__C.TRAIN.RPN_MIN_SIZE = 16

__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
# __C.TEST.RPN_PRE_NMS_TOP_N = 12000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# __C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

###### RCNN OPTION ######
# ROI POOLING
__C.POOLING_SIZE = 7

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1


# RCNN inference option

__C.TEST.BBOX_REG = True

__C.TEST.NMS = 0.3









