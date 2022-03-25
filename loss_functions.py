from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import cv2


# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

# Function to calculate boundary uncertainty
def border_uncertainty(seg, alpha = 0.9, beta = 0.1):
    """
    Parameters
    ----------
    alpha : float, optional
        controls certainty of ground truth inner borders, by default 0.9.
        Higher values more appropriate when over-segmentation is a concern
    beta : float, optional
        controls certainty of ground truth outer borders, by default 0.1
        Higher values more appropriate when under-segmentation is a concern
    """

    res = np.zeros_like(seg)
    check_seg = seg.astype(np.bool)

    if check_seg.any():
        kernel = np.ones((3,3),np.uint8)
        im_erode = cv2.erode(seg[:,:,:,1],kernel,iterations = 1)
        im_dilate = cv2.dilate(seg[:,:,:,1],kernel,iterations = 1)
        # compute inner border and adjust certainty with alpha parameter
        inner = seg[:,:,:,1] - im_erode
        inner = alpha * inner
        # compute outer border and adjust certainty with beta parameter
        outer = im_dilate - seg[:,:,:,1]
        outer = beta * outer
        # combine adjusted borders together with unadjusted image
        combined = inner + outer + im_erode
        combined = np.expand_dims(combined,axis=-1)

        res = np.concatenate([1-combined, combined],axis=-1)

        return res
    else:
        return res

# Enables batch processing of boundary uncertainty
def border_uncertainty_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([border_uncertainty(y) for y in y_true_numpy]).astype(np.float32)


# Dice loss
def dice_loss(y_true, y_pred, boundary=True, smooth=0.00001):

    # Identify axis
    axis = identify_axis(y_true.get_shape())

    if boundary:
        y_true = tf.py_function(func=border_uncertainty_batch, inp=[y_true], Tout=tf.float32)

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)

    return 1 - dice

# Dice and cross entropy loss
def dice_ce_loss(y_true, y_pred, boundary=True, smooth=0.00001):
    dice_loss = dice_loss(y_true, y_pred, boundary=boundary, smooth=smooth)
    crossentropy = K.categorical_crossentropy(y_true, y_pred)
    crossentropy = K.mean(crossentropy)
    return dice_loss + crossentropy


# Asymmetric Focal Tversky loss
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75, boundary=True):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """
    def loss_function(y_true, y_pred):

        if boundary:
            y_true = tf.py_function(func=border_uncertainty_batch,inp=[y_true],Tout=tf.float32)

        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        axis = identify_axis(y_true.get_shape())
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma) 

        # Average class scores
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))
        return loss

    return loss_function

# Asymmetric Focal loss
def asymmetric_focal_loss(delta=0.7, gamma=2., boundary=True):

    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    """

    def loss_function(y_true, y_pred):
        if boundary:
            y_true = tf.py_function(func=border_uncertainty_batch,inp=[y_true],Tout=tf.float32)

        axis = identify_axis(y_true.get_shape())  

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        #calculate losses separately for each class, only suppressing background class
        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

# Asymmetric Unified Focal loss
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5, boundary=True):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma, boundary=boundary)(y_true,y_pred)
      asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma, boundary=boundary)(y_true,y_pred)
      if weight is not None:
        return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl

    return loss_function

# Symmetric Unified Focal loss
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5, boundary=True):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def loss_function(y_true,y_pred):
      symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma, boundary=boundary)(y_true,y_pred)
      symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma, boundary=boundary)(y_true,y_pred)
      if weight is not None:
        return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)  
      else:
        return symmetric_ftl + symmetric_fl

    return loss_function
