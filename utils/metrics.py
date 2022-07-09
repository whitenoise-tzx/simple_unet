import numpy as np
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    sum_n_ii = 0
    sum_t_i = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i
    # print("pixel accuracy = %f" %pixel_accuracy_)
    return pixel_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = n_ii / (t_i + n_ij - n_ii)
    mean_IU_ = np.sum(IU) / n_cl_gt
    # print("miou = %f" % mean_IU_)
    return mean_IU_


def convert_binary(mask):
    h, w = mask.shape
    mask_b = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            if mask[row, col] != 2:
                mask_b[row, col] = 0
            else:
                mask_b[row, col] = 2
    return mask_b

# General util function to get the boundary of a binary mask.
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def boundary_iou(gt,pred):
    gt = convert_binary(gt)
    pred = convert_binary(pred)
    check_size(pred, gt)
    cl, n_cl = union_classes(pred, gt)
    _, n_cl_gt = extract_classes(gt)
    eval_mask, gt_mask = extract_both_masks(pred, gt, cl, n_cl)
    IU = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = n_ii / (t_i + n_ij - n_ii)
    #boundary_iou = np.sum(IU) / n_cl_gt
    #print("boundary_iou = %f" % mean_IU_)
    if len(IU) == 2:
        return IU[1]
    else:
        return 0.0


def boundary_iou_ori(gt, dt):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    #gt_boundary = mask_to_boundary(gt, dilation_ratio)
    #dt_boundary = mask_to_boundary(dt, dilation_ratio)
    gt_boundary = convert_binary(gt)
    dt_boundary = convert_binary(dt)
    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(gt_boundary)
    plt.subplot(1, 2, 2)
    plt.imshow(dt_boundary)
    plt.show()
    """
    intersection = np.sum(np.logical_and(dt_boundary, gt_boundary))
    gt_i = np.sum(gt_boundary)
    dt_ij = np.sum(dt_boundary)
    boundary_iou = intersection / (gt_i + dt_ij - intersection)
    """
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    """
    #print('boundary_iou=%f'%boundary_iou)
    return boundary_iou


def computeQualityMeasures(pred, gt):
    lP=convert_binary(pred)
    lT=convert_binary(gt)
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    avgHausdorff = hausdorffcomputer.GetAverageHausdorffDistance()

    Hausdorff = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    dice = dicecomputer.GetDiceCoefficient()

    return avgHausdorff, Hausdorff, dice



'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm) #cl=[0 1 2]
    n_cl = len(cl) #n_cl=3
    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width

def check_size(eval_segm, gt_segm):

    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    #print(h_e, w_e)
    #print(h_g, w_g)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)