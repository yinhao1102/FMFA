import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_a_sdm(image_fetures, text_fetures, pid, logit_scale, alpha_i2t, alpha_t2i, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    i2t_sim_max_idx = image_proj_text.argmax(dim=1)
    t2i_sim_max_idx = text_proj_image.argmax(dim=1)
    label_max_idx = labels_distribute.argmax(dim=1)

    i2t_mask = i2t_sim_max_idx != label_max_idx
    diff_i2t = i2t_cosine_theta[torch.arange(i2t_cosine_theta.shape[0]), i2t_sim_max_idx] - i2t_cosine_theta[torch.arange(i2t_cosine_theta.shape[0]), label_max_idx]
    diff_i2t[~i2t_mask] = 0
    diff_i2t = diff_i2t * alpha_i2t + 1

    t2i_mask = t2i_sim_max_idx != label_max_idx 
    diff_t2i = t2i_cosine_theta[torch.arange(t2i_cosine_theta.shape[0]), t2i_sim_max_idx] - t2i_cosine_theta[torch.arange(t2i_cosine_theta.shape[0]), label_max_idx]
    diff_t2i[~t2i_mask] = 0
    diff_t2i = diff_t2i * alpha_t2i + 1

    loss = torch.mean(diff_i2t *torch.sum(i2t_loss, dim=1)) + torch.mean(diff_t2i *torch.sum(t2i_loss, dim=1))

    return loss

def compute_infonce_loss_cosine(X, C, margin=0.2, sim_temperature=0.1, efa_tempurature=1.0):
    X_norm = F.normalize(X, p=2, dim=-1)
    C_norm = F.normalize(C, p=2, dim=-1)
    X_norm = X_norm.unsqueeze(1)
    C_norm = C_norm.unsqueeze(0)

    sim_xc = torch.matmul(X_norm, C_norm.transpose(-2,-1))  # [B, B, L/N, L]
    # sim_xc = torch.einsum('blc,bjc->bjbl', X_norm, C_norm)

    max_pooled_sim, _ = torch.max(sim_xc, dim=-1)  # [B, L]
    sims = torch.logsumexp(max_pooled_sim/sim_temperature, dim=-1)

    sims = F.normalize(sims, p=2, dim=-1)

    ## cost of image retrieval
    img_ret = sims-sims.diag().expand_as(sims).t()+margin
    img_ret[torch.eye(sims.size(0))>.5] = 0
    # cost_im = torch.log(torch.sum(torch.exp(img_ret/ efa_tempurature ),dim=1))
    max_img_ret = torch.max(img_ret, dim=0, keepdim=True)[0]
    stabilized_img_ret = img_ret - max_img_ret
    cost_im = torch.log(torch.sum(torch.exp(stabilized_img_ret / efa_tempurature), dim=0)) + max_img_ret

    ## cost of text retrieval
    txt_ret = sims-sims.diag().expand_as(sims)+margin
    txt_ret[torch.eye(sims.size(0))>.5] = 0
    # cost_s = torch.log(torch.sum(torch.exp(txt_ret/ efa_tempurature ),dim=0))
    max_txt_ret = torch.max(txt_ret, dim=0, keepdim=True)[0]
    stabilized_txt_ret = txt_ret - max_txt_ret
    cost_s = torch.log(torch.sum(torch.exp(stabilized_txt_ret / efa_tempurature), dim=0)) + max_txt_ret

    return cost_s.mean() + cost_im.mean()

def compute_efa(text_feat, image_feat, multimodal_feat, margin_t2e, margin_i2e):
    loaa_ac=compute_infonce_loss_cosine(text_feat, multimodal_feat, margin_t2e)
    loaa_bc=compute_infonce_loss_cosine(image_feat, multimodal_feat, margin_i2e)

    return loaa_ac+loaa_bc

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

