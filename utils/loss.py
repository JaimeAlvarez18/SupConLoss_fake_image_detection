import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Loss (SupCon).

    Reference:
        Khosla et al., "Supervised Contrastive Learning" (https://arxiv.org/pdf/2004.11362.pdf)

    This implementation also degrades to the SimCLR unsupervised loss when labels do not provide positives.
    The loss expects `features` shaped [bsz, n_views, ...] and `labels` shaped [bsz].
    - `features` is typically a batch of embedding vectors produced by a backbone.
    - `n_views` is the number of augmented views per example.

    Temperature is a learnable parameter stored as a torch.nn.Parameter so it will be part of optimizer state.
    """
    def __init__(self, t_0=0.07, eps=1e-8):
        super(SupConLoss, self).__init__()
        # Learnable temperature (initialized to t_0)
        self.temperature = torch.nn.Parameter(torch.tensor([t_0]))
        # Small epsilon for numerical stability in log/sum operations
        self.epsilon = eps

    def forward(self, features, labels):
        """Compute the supervised contrastive loss.

        Args:
            features: Tensor of shape [bsz, n_views, ...] (embeddings per view).
            labels: Tensor of shape [bsz] containing class labels.

        Returns:
            loss: Scalar tensor with the computed SupCon loss.
        """
        batch_size = features.shape[0]

        # Basic checks: require at least 3 dimensions (bsz, n_views, feat_dim)
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        # If features have extra spatial dimensions, flatten them
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        # Ensure labels shape is [bsz, 1]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # Mask where mask[i,j] = 1 if labels[i] == labels[j], else 0
        mask = torch.eq(labels, labels.T).float().to(features.device)

        views = features.shape[1]  # number of views / augmentations per example

        # Concatenate views into a single axis: [bsz * views, feat_dim]
        full_features = torch.cat(torch.unbind(features, dim=1), dim=0).to(features.device)

        # Compute pairwise cosine similarities (dot of normalized vectors) and scale by temperature
        temperature = self.temperature.clamp(min=1e-4, max=1e2).to(features.device)
        anchor_dot_contrast = torch.matmul(
            F.normalize(full_features, dim=1),
            F.normalize(full_features, dim=1).T
        ) / temperature

        # Delegate to helper to compute stable log-prob-based loss from dot products
        loss = self._loss_from_dot(anchor_dot_contrast, mask, views, batch_size)

        return loss

    def _loss_from_dot(self, anchor_dot_contrast, mask, views, batch_size):
        """Compute supervised contrastive loss given precomputed dot products.

        Steps:
        1. Numerically stabilize logits by subtracting max per row.
        2. Mask out self-contrast terms.
        3. Compute log-probabilities and average log-prob over positives.
        4. Return the negative mean log-prob (loss).
        """
        # Numerical stability: subtract row-wise max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile the mask for the expanded views: mask shape becomes [bsz*views, bsz*views]
        mask = mask.repeat(views, views)

        # Mask out self-contrast cases (i == j)
        logits_mask = 1 - torch.eye(views * batch_size, device=mask.device)
        mask = mask * logits_mask

        # Exponentiate logits, mask self-contrasts, compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # Mean log-likelihood over positive pairs for each anchor
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.epsilon)

        # Reshape to (views, batch_size) and average
        loss = - mean_log_prob_pos.view(views, batch_size).mean()

        return loss