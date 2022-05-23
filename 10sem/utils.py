import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def train(model, optimizer, t_train, snapdataloaders, deltas, items_embs, users_sample_size, users_batch_size, neg_sampler):
    """Trains model.
    """
    device = next(model.parameters()).device
    train_loss_vals = np.empty(t_train)
    model.train()
    for t in range(t_train):
        loss = torch.tensor(0., device=device)
        model.get_graph_item_embs(snapdataloaders[t], out=items_embs)
        users_inds_batches = torch.split(
            torch.randperm(model.num_users, dtype=torch.int32)[:users_sample_size],
            users_batch_size
        )
        for users_inds_batch in users_inds_batches[:-1]:
            users_batch_embs = model.get_user_embs(deltas[:t+1], items_embs, users_inds_batch)
            pos_g = deltas[t+1].subgraph({"item": deltas[t+1].nodes("item"), "user": users_inds_batch},
                                         output_device=device)
            neg_g = dgl.heterograph({("item", "iu", "user"): neg_sampler(pos_g, pos_g.edges("eid"))},
                                    {"item": pos_g.num_nodes("item"), "user": pos_g.num_nodes("user")},
                                    device=device)
            pos_score = model.predictor(pos_g, items_embs, users_batch_embs)
            neg_score = model.predictor(neg_g, items_embs, users_batch_embs)
            loss += compute_loss(pos_score, neg_score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        items_embs.detach_()
        train_loss_vals[t] = loss.item() / (len(users_inds_batches)  - 1)
    return train_loss_vals.mean()


def test(model, t_train, snapdataloaders, deltas, items_embs, users_test_inds_batch, neg_sampler):
    """Validates model.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        model.get_graph_item_embs(snapdataloaders[t_train], out=items_embs)
        users_test_batch_embs = model.get_user_embs(deltas[:t_train+1], items_embs, users_test_inds_batch)
        pos_g = deltas[t_train+1].subgraph({"item": deltas[t_train+1].nodes("item"), "user": users_test_inds_batch},
                                           output_device=device)
        neg_g = dgl.heterograph({("item", "iu", "user"): neg_sampler(pos_g, pos_g.edges("eid"))},
                                {"item": pos_g.num_nodes("item"), "user": pos_g.num_nodes("user")},
                                device=device)
        pos_score = model.predictor(pos_g, items_embs, users_test_batch_embs)
        neg_score = model.predictor(neg_g, items_embs, users_test_batch_embs)
        test_loss = compute_loss(pos_score, neg_score)
    return test_loss.item()


def inference(model, t_train, snapdataloaders, deltas, items_embs, users_test_inds, users_test_true, item_enc):
    model.eval()
    with torch.no_grad():
        model.get_graph_item_embs(snapdataloaders[t_train], out=items_embs)
        users_test_embs = model.get_user_embs(deltas[:t_train+1], items_embs, users_test_inds)
        users_test_pred = model.predict_purchases(items_embs, users_test_embs, num_preds=12, item_enc=item_enc)
        test_metric_val = mean_average_precision(users_test_true, users_test_pred)
    return test_metric_val


def show(epoch, epochs, train_loss_vals, test_loss_vals, test_metric_vals):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f'Epoch: {epoch}/{epochs}; Loss: {train_loss_vals[-1]:.4f};'
                 f' Test Loss: {test_loss_vals[-1] if len(test_loss_vals) else 0:.4f};'
                 f' Test Metric: {test_metric_vals[-1] if len(test_metric_vals) else 0:.6f}')
    axs[0].plot(train_loss_vals, label="train loss")
    axs[1].plot(test_loss_vals, label="test loss")
    axs[2].plot(test_metric_vals, label="test metric")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]),
                        torch.zeros(neg_score.shape[0])]).to(pos_score.device)
    return nn.BCEWithLogitsLoss()(scores, labels)


def precision_at_k(y_true, y_pred, k=12):
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k


def rel_at_k(y_true, y_pred, k=12):
    if y_pred[k - 1] in y_true:
        return 1
    else:
        return 0


def average_precision_at_k(y_true, y_pred, k=12):
    ap = 0.0
    for i in range(1, k + 1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)

    return ap / min(k, len(y_true))


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k

    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])


def get_users_test(t_train, users_test_size, users_test_batch_size):
    users_test_inds = deltas[t_train + 1].edges()[1].unique()
    shuffled_inds = torch.randperm(users_test_inds.size(0))
    users_test_inds_batch = users_test_inds[shuffled_inds[:users_test_batch_size]]
    users_test_inds = users_test_inds[shuffled_inds[:users_test_size]]

    max_num_items = -np.inf
    users_test_true = list()
    test_week_edf = edf[edf.week == t_train + 1]
    for user_id in user_enc.inverse_transform(users_test_inds):
        user_true = test_week_edf[test_week_edf.customer_id == user_id].article_id.values
        user_true = np.unique(user_true)
        users_test_true.append(user_true)
        if user_true.shape[0] > max_num_items:
            max_num_items = user_true.shape[0]
    for i in range(len(users_test_true)):
        user_true = users_test_true[i]
        if user_true.shape[0] < max_num_items:
            padding = np.full((max_num_items - user_true.shape[0]), user_true[-1])
            user_true = np.concatenate((user_true, padding))
            users_test_true[i] = user_true
    users_test_true = np.array(users_test_true)
    return users_test_inds_batch, users_test_inds, users_test_true
