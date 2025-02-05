import sys
import torch

def compute_entropy(view1, view2, args):
    bn, c = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == c)
    p_j_k = torch.mm(view1.t(), view2)

    p_j_k = (p_j_k + p_j_k.t()) / 2.    # symmetrise
    p_j_k = p_j_k / p_j_k.sum()         # normalise
    p_j = p_j_k.sum(dim = 1)
    p_k = p_j_k.sum(dim = 0)

    # h_k_j = p_j_k / p_j
    h_j_k = p_j_k / p_k
    h_j   = p_j.view(c, 1).expand(c, c)
    h_k   = p_k.view(1, c).expand(c, c)


    return h_j_k, h_j, h_k


def Classifying_Consensus_Loss(z1, z2, args, alpha= 10.0, beta = 10.0, gamma = 10, EPS=sys.float_info.epsilon):


    """Classifying Consensus Learning loss for minimizing conditional entropy and maximizng entropy """
    _, c = z1.size()
    # Compute entropy
    h_j_k, h_j, h_k = compute_entropy(z1, z2, args)
    # from ipdb import set_trace
    # set_trace()
    assert (h_j_k.size() == (c, c))

    h_j_k = torch.where(h_j_k < EPS, torch.tensor([EPS], device = h_j_k.device), h_j_k)
    h_k = torch.where(h_k < EPS, torch.tensor([EPS], device = h_k.device), h_k)
    h_j = torch.where(h_j < EPS, torch.tensor([EPS], device = h_j.device), h_j)


    loss = - h_j_k*h_k * (alpha * torch.log(h_j_k) - beta * torch.log(h_k) - gamma * torch.log(h_j))


    loss = loss.sum()

    return loss