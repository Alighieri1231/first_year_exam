import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from src.utils.utils import *


def BCELogitsLoss(y_hat, y, weight=None):
    return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=None)
    # return F.binary_cross_entropy(y_hat, y)


def BCEDiceLoss(y_hat, y, weight=0.1, device="cuda"):
    # bce_loss = F.binary_cross_entropy(y_hat, y)
    if y_hat.shape[1] == 1:
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
    else:
        y_hat = y_hat[:, 1, ...]
        y = y[:, 1, ...]
        bce_loss = F.binary_cross_entropy(y_hat, y)
    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    loss = bce_loss * weight + dice_loss * (1 - weight)
    # loss = bce_loss + dice_loss

    return loss


# cross entropy
def CrossEntropyLoss(y_hat, y):
    return F.cross_entropy(y_hat, y)


def DiceLoss(y_hat, y):
    if y_hat.shape[1] == 1:
        y_hat = torch.sigmoid(y_hat)
    else:
        # extract only second channel (foreground)
        y_hat = y_hat[:, 1, ...]
        y = y[:, 1, ...]

    _, dice_loss = utils.dice_coeff_batch(y_hat, y)
    return dice_loss


def FocalLossb(y_hat, y, alpha=0.5, gamma=2, logits=False, reduce=True):
    # if logits:
    #     BCE_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
    # else:
    #     BCE_loss = F.binary_cross_entropy(y_hat, y, reduction='none')
    BCE_loss = F.binary_cross_entropy(y_hat, y, reduction="none")
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss


def FocalLoss_multiclass(y_hat, y, alpha=0.5, gamma=2):
    """
    Optimized Focal Loss for multiclass classification.

    Args:
        y_hat (torch.Tensor): Logits de salida del modelo con forma [batch_size, num_classes, ...].
        y (torch.Tensor): Etiquetas verdaderas con forma [batch_size, ...] (no one-hot encoded).
        alpha (float): Peso para la clase positiva.
        gamma (float): Parámetro de ajuste de focalidad.

    Returns:
        torch.Tensor: Pérdida promedio calculada.
    """
    # Log-Softmax y probabilidades verdaderas
    log_probs = F.log_softmax(y_hat, dim=1)  # [batch_size, num_classes, ...]
    probs_true = torch.exp(
        log_probs.gather(dim=1, index=y.unsqueeze(1)).squeeze(1)
    )  # [batch_size, ...]

    # Cálculo de Focal Loss
    focal_weight = alpha * (1 - probs_true) ** gamma
    ce_loss = -log_probs.gather(dim=1, index=y.unsqueeze(1)).squeeze(
        1
    )  # Cross-entropy loss por muestra
    focal_loss = focal_weight * ce_loss

    return focal_loss.mean()


class FocalLoss(nn.Module):
    # WC: alpha is weighting factor. gamma is focusing parameter
    def __init__(self, gamma=0, alpha=None, size_average=True):
        # def __init__(self, gamma=2, alpha=0.25, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def FocalLoss_weights(y_hat, y, alpha=0.5, gamma=2, logits=False, reduce=True):
    weights = [1.2273, 5.4000]
    y_aux = torch.tensor(y, dtype=torch.int)
    ce_loss = F.cross_entropy(y_hat, y, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = (weights[y_aux.item()] * (1 - pt) ** gamma * ce_loss).mean()
    return loss


def SigmoidFocalLoss(y_hat, y):
    F_loss = sigmoid_focal_loss(y_hat, y, alpha=0.25, gamma=2, reduction="mean")
    return F_loss


# def NLLLoss(y_hat, y):
#     # loss = torch.nn.NLLLoss()
#     # m = torch.nn.LogSoftmax(dim=1) # assuming tensor is of size N x C x height x width, where N is the batch size.
#     loss = F.nll_loss(F.log_softmax(y_hat), y)
#     return loss

# class WeightedFocalLoss(torch.nn.Module):
#     "Non weighted version of Focal Loss"
#     "https://amaarora.github.io/2020/06/29/FocalLoss.html"
#     def __init__(self, alpha=.25, gamma=2):
#         super(WeightedFocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()


def TopologicalPoolingLoss(y_hat, y, combined=1, device="cuda"):
    def loss_per_batch(y_hat, y):
        axial_pool = nn.MaxPool3d(kernel_size=(1, 1, 128))
        sagittal_pool = nn.MaxPool3d(kernel_size=(128, 1, 1))
        coronal_pool = nn.MaxPool3d(kernel_size=(1, 128, 1))

        # Aplicar pooling
        P_axial = axial_pool(y_hat)
        P_sagittal = sagittal_pool(y_hat)
        P_coronal = coronal_pool(y_hat)

        P_axial = P_axial[:, :, :, 0]
        P_sagittal = P_sagittal[:, 0, :, :]
        P_coronal = P_coronal[:, :, 0, :]

        # Aplicar pooling
        G_axial = axial_pool(y)
        G_sagittal = sagittal_pool(y)
        G_coronal = coronal_pool(y)

        G_axial = G_axial[:, :, :, 0]
        G_sagittal = G_sagittal[:, 0, :, :]
        G_coronal = G_coronal[:, :, 0, :]

        # kernel_sizes = [4, 5, 8, 10, 20]  # [2, 4, 8, 12, 15, 20]
        kernel_sizes = [2, 4, 8, 16, 32]
        loss_values = []

        for k_size in kernel_sizes:
            pool_2d = nn.MaxPool2d(kernel_size=k_size)
            P_topo_k = pool_2d(P_axial) + pool_2d(P_sagittal) + pool_2d(P_coronal)
            G_topo_k = pool_2d(G_axial) + pool_2d(G_sagittal) + pool_2d(G_coronal)
            # Calcular la diferencia absoluta

            diff = torch.abs(
                ((G_topo_k) - (P_topo_k))
            )  # Obtengo cuantos pixeles no coinciden con GT en los 3 planos
            loss_k_mean = torch.mean(
                diff
            )  # Promedio de los errores para cada tamaño de k
            loss_values.append(loss_k_mean)
        # Promediar sobre todos los tamaños de kernels
        L_topo = torch.mean(torch.stack(loss_values))
        return L_topo

    # print(y_hat.shape, y.shape)
    if y_hat.shape[1] == 1:
        bce_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        y_hat = torch.sigmoid(y_hat)
    else:
        y_hat = y_hat[:, 1, ...]
        y = y[:, 1, ...]
        # expanding the dimensions of y_hat and y
        y_hat = y_hat.unsqueeze(1)
        y = y.unsqueeze(1)
        bce_loss = F.binary_cross_entropy(y_hat, y)
    # Init dice score for batch (GPU ready)
    if y_hat.is_cuda:
        l_topo_val = torch.FloatTensor(1).cuda(device="cuda").zero_()
    else:
        l_topo_val = torch.FloatTensor(1).zero_()
    # print(y_hat.shape, y.shape)
    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(y_hat, y)):
        l_topo_val += loss_per_batch(inputs[0], inputs[1])

    # Return the mean Dice coefficient over the given batch
    l_topo_batch = l_topo_val / (pair_idx + 1)

    # _, dice_loss = utils.dice_coeff_batch(y_hat, y)
    # lambda_val=1
    # L_topo=l_topo_batch+lambda_val*(dice_loss)

    # weight = 0.5
    # L_topo = l_topo_batch*weight + dice_loss*(1-weight)

    _, dice_loss = utils.dice_coeff_batch(y_hat, y, device)
    weight = 0.33
    # L_topo = l_topo_batch * weight + bce_loss * weight + dice_loss * (weight)
    lamb = 1
    if combined == 1:
        L_topo = l_topo_batch * weight + dice_loss * (1 - weight)
    else:
        L_topo = l_topo_batch

    return L_topo


class TopologicalPoolingLossClass(nn.Module):
    def __init__(self, start_channel=1, kernel_list=None, stride_list=None):
        """Initializes the TopologicalPoolingLoss class."""
        super().__init__()
        self.start_channel = start_channel
        self.kernel_list = kernel_list or [4, 5, 8, 10, 20]
        self.stride_list = stride_list or self.kernel_list

    def forward(self, input, target):
        """Computes the topological pooling loss for the input and target tensors."""
        if input.dim() != target.dim():
            raise ValueError("'input' and 'target' have different number of dimensions")
        if input.dim() not in (4, 5):
            raise ValueError("'input' and 'target' must be 4D or 5D tensors")
        per_channel_topology_component = compute_per_channel_topology_component(
            input, target, self.start_channel, self.kernel_list, self.stride_list
        )
        return per_channel_topology_component


def project_pooling_3d_tensor(input_tensor, kernel_size):
    """Applies max pooling on the 3D tensor with the specified kernel size."""
    project_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
    return project_pooling(input_tensor)


def topology_pooling_2d_tensor(input_tensor, kernel, stride):
    """Applies max pooling on the 2D tensor with the specified kernel size and stride."""
    abstract_2d_pooling = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    abstract_pooling = abstract_2d_pooling(input_tensor)
    return abstract_pooling


def topological_pooling(input_tensor, kernel, stride, dim):
    """Performs topological pooling on the input tensor."""
    if input_tensor.dim() == 5:  # 3D volumes
        projection_kernels = [
            (1, 1, input_tensor.size(4)),
            (input_tensor.size(2), 1, 1),
            (1, input_tensor.size(3), 1),
        ]
        input_project_pooling_3d_tensor = project_pooling_3d_tensor(
            input_tensor, kernel_size=projection_kernels[dim]
        )
        if dim == 0:
            squeeze_dim = 4
        else:
            squeeze_dim = 1
        input_project_pooling_3d_tensor = input_project_pooling_3d_tensor.squeeze(
            dim + squeeze_dim
        )
    elif input_tensor.dim() == 4:  # 2D images
        input_project_pooling_3d_tensor = input_tensor
    else:
        raise ValueError("'input_tensor' must be 4D or 5D tensors")
    input_2d_pooling = topology_pooling_2d_tensor(
        input_project_pooling_3d_tensor, kernel=kernel, stride=stride
    )
    return input_2d_pooling


def compute_per_channel_topology_component(
    input, target, start_channel, kernel_list, stride_list
):
    """Computes the per-channel topology component of the input and target tensors."""
    assert input.size() == target.size(), (
        "'input' and 'target' must have the same shape"
    )
    num_channels = input.size(1)
    num_dims = input.dim() - 2  # Calculate the number of dimensions: 3 for 3D, 2 for 2D
    difference_ks_list = []
    for kernel, stride in zip(kernel_list, stride_list):
        pooling_diff = []
        for dim in range(
            num_dims
        ):  # Change the loop range to accommodate 2D and 3D tensors
            pred_pooling = topological_pooling(
                input, kernel=kernel, stride=stride, dim=dim
            )
            label_pooling = topological_pooling(
                target, kernel=kernel, stride=stride, dim=dim
            )
            channel_pooling_diff = []
            for channel in range(
                start_channel, num_channels
            ):  # start from 1 to ignore the background channel.
                sum_pred_pooling = torch.sum(pred_pooling, dim=(-2, -1))[
                    :, channel, ...
                ]
                sum_label_pooling = torch.sum(label_pooling, dim=(-2, -1))[
                    :, channel, ...
                ]
                difference = torch.abs(sum_pred_pooling - sum_label_pooling)
                channel_pooling_diff.append(difference)
            pooling_diff.append(torch.mean(torch.stack(channel_pooling_diff)))
        difference_ks_list.append(torch.mean(torch.stack(pooling_diff)))
    return torch.mean(torch.stack(difference_ks_list))


class DiceLossOriginal(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLossOriginal, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        """
        Compute Dice Loss for multi-channel input (e.g., 2 channels for background and foreground).

        Args:
        - input: Predicted probabilities (after softmax) of shape [batch_size, 2, ...].
        - target: Ground truth one-hot encoded of shape [batch_size, 2, ...].

        Returns:
        - Dice loss (scalar).
        """
        # Use only the foreground channel for Dice Loss computation
        input_fg = input[:, 1, ...]  # Channel for foreground
        target_fg = target[:, 1, ...]  # Channel for foreground

        # Flatten the tensors
        input_fg = input_fg.contiguous().view(input_fg.size(0), -1)
        target_fg = target_fg.contiguous().view(target_fg.size(0), -1)

        # Compute intersection and sums
        intersection = torch.sum(input_fg * target_fg, dim=1)
        input_sum = torch.sum(input_fg**2, dim=1)
        target_sum = torch.sum(target_fg**2, dim=1)

        # Dice coefficient
        dice_coeff = (2 * intersection + self.eps) / (input_sum + target_sum)

        # Dice loss
        dice_loss = 1 - dice_coeff.mean()  # Mean over the batch
        return dice_loss


class BCEDiceLossOriginal(nn.Module):
    def __init__(self, dice_eps=1e-6, weight=0.1):
        """
        Combina Binary Cross-Entropy (BCE) y Dice Loss, solo para el foreground.

        Args:
        - dice_eps (float): Pequeño valor para estabilizar el denominador del Dice Loss.
        - weight (float): Peso para la BCE Loss en la combinación final.
        """
        super(BCEDiceLossOriginal, self).__init__()
        self.dice_loss = DiceLossOriginal(eps=dice_eps)
        self.bce = nn.BCELoss()  # BCE solo para el foreground
        self.weight = weight

    def forward(self, input, target):
        """
        Calcula la pérdida combinada BCE + Dice.

        Args:
        - input: Predicciones del modelo (probabilidades después de Sigmoid o Softmax) [batch_size, 2, ...].
        - target: Máscara de verdad (one-hot encoded) [batch_size, 2, ...].

        Returns:
        - Pérdida combinada BCE + Dice (scalar).
        """
        # Usar solo el canal de foreground
        input_fg = input[:, 1, ...]  # Canal foreground
        target_fg = target[:, 1, ...]  # Canal foreground

        # Calcular BCE solo para el foreground
        bce_loss = self.bce(input_fg, target_fg)

        # Calcular Dice Loss (solo foreground)
        dice_loss = self.dice_loss(input, target)

        # Combinar BCE y Dice Loss
        combined_loss = self.weight * bce_loss + (1 - self.weight) * dice_loss
        return combined_loss


class TopologyDiceLossOriginal(nn.Module):
    def __init__(self, dice_eps=1e-6, weight=1):
        """
        Combina Binary Cross-Entropy (BCE) y Dice Loss, solo para el foreground.

        Args:
        - dice_eps (float): Pequeño valor para estabilizar el denominador del Dice Loss.
        - weight (float): Peso para la BCE Loss en la combinación final.
        """
        super(TopologyDiceLossOriginal, self).__init__()
        self.dice_loss = DiceLossOriginal(eps=dice_eps)
        self.topology_loss = TopologicalPoolingLossClass()
        self.weight = weight

    def forward(self, input, target):
        """
        Calcula la pérdida combinada BCE + Dice.

        Args:
        - input: Predicciones del modelo (probabilidades después de Sigmoid o Softmax) [batch_size, 2, ...].
        - target: Máscara de verdad (one-hot encoded) [batch_size, 2, ...].

        Returns:
        - Pérdida combinada BCE + Dice (scalar).
        """
        # topology_loss = self.topology_loss(input, target)
        # probar con funcion
        topology_loss = TopologicalPoolingLoss(input, target, combined=0)

        # Calcular Dice Loss (solo foreground)
        dice_loss = self.dice_loss(input, target)

        # Combinar BCE y Dice Loss
        combined_loss = self.weight * topology_loss + self.weight * dice_loss
        return combined_loss


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        # máscara de píxeles válidos
        target_mask = (target >= 0) & (target != self.ignore_label)
        target = target[target_mask]
        if not target.dim():
            # devuelve un tensor en el mismo dispositivo
            return torch.zeros(1, device=predict.device)

        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)]
        predict = predict.view(-1, c)

        reduction = "mean" if self.size_average else "sum"
        loss = F.cross_entropy(predict, target, weight=weight, reduction=reduction)
        return loss


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        n, c, h, w = predict.size()
        print (f"predict.shape: {predict.shape}, target.shape: {target.shape}")
        target_mask = (target >= 0) & (target != self.ignore_label)
        target = target[target_mask]
        if not target.dim():
            return torch.zeros(1, device=predict.device)

        predict = predict[target_mask]

        reduction = "mean" if self.size_average else "sum"
        loss = F.binary_cross_entropy_with_logits(
            predict, target, weight=weight, reduction=reduction
        )
        return loss
