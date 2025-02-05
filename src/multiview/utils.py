import logging
import numpy as np
import math
import torch
import ipdb 
import time as Time

def get_logger(filename):
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# def next_batch(X11, X12, X21, X22, batch_size):
#     """Return data for next batch"""
#     tot1 = X11.shape[0]
#     tot2 = X12.shape[0]
#     total1 = math.ceil(tot1 / batch_size)
#     total2 = math.ceil(tot2 / batch_size)
#     # batch_x12 = torch.tensor([]).float().to(device)
#     # batch_x22 = torch.tensor([]).float().to(device)
#     for i in range(int(total1)):
#         start_idx1 = i * batch_size
#         end_idx1 = (i + 1) * batch_size
#         end_idx1 = min(tot1, end_idx1)
#         end_idx2 = (i + 1) * batch_size
#         end_idx2 = min(tot2, end_idx2)
#         batch_x11 = X11[start_idx1: end_idx1, ...]
#         batch_x21 = X21[start_idx1: end_idx1, ...]
#         batch_x12 = X12[start_idx1: end_idx1, ...]
#         batch_x22 = X22[start_idx1: end_idx1, ...]
#         yield (batch_x11, batch_x12, batch_x21, batch_x22, (i + 1))

def next_batch(X1, X2, X3, batch_size):
    """Return data for next batch"""
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size)
    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        batch_x3 = X3[start_idx: end_idx, ...]

        yield (batch_x1, batch_x2,batch_x3, (i + 1))


def cal_std(logger, *arg):
    """Return the average and its std"""
    if len(arg) == 3:
        logger.info('ACC:'+ str(arg[0]))
        logger.info('NMI:'+ str(arg[1]))
        logger.info('ARI:'+ str(arg[2]))
        output = """ ACC {:.2f} std {:.2f} NMI {:.2f} std {:.2f} ARI {:.2f} std {:.2f}""".format(np.mean(arg[0]) * 100,
                                                                                                 np.std(arg[0]) * 100,
                                                                                                 np.mean(arg[1]) * 100,
                                                                                                 np.std(arg[1]) * 100,
                                                                                                 np.mean(arg[2]) * 100,
                                                                                                 np.std(arg[2]) * 100)
    elif len(arg) == 1:
        logger.info(arg)
        output = """ACC {:.2f} std {:.2f}""".format(np.mean(arg) * 100, np.std(arg) * 100)
    logger.info(output)



def normalize(x):
    """Normalize"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def drop_feature(x, drop_prob):
    # ipdb.set_trace()

    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    # out=sum(X)
    return x



def tensor_shuffle(x): 
    # ipdb.set_trace() 
    """  
    Shuffles the rows of a 2D tensor.  
  
    Args:  
        tensor (torch.Tensor): The input tensor.  
  
    Returns:  
        torch.Tensor: The shuffled tensor.  
    """  
    # Creates a randomly arranged index
    random_int_sequences = [torch.randperm(x.shape[1]) for _ in range(x.shape[0])]  
    idx = torch.stack(random_int_sequences, dim=0).cuda()  
    # Apply this random arrangement on each line
    shuffled_x = x.gather(1, idx)  
    return shuffled_x