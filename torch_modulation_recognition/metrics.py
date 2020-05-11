import torch


@torch.no_grad()
def accuracy_topk(y_pred: torch.Tensor, y_true: torch.Tensor, k: int):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    correct_k.mul_(100.0 / batch_size)
    return correct_k