import torch
import scipy.ndimage as nd


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


def rotate_img(x, deg):
    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()


def hits(y, y_pred, k, df, batch_h, batch_r, filtered=False):
    if filtered:
        tmp = torch.argsort(y_pred, dim=1, descending=True).numpy()
        tmp2 = []
        for idx, (r, tail, rel) in enumerate(zip(tmp, batch_h, batch_r)):
            ent = df[(df['e2'] == tail) & (df['rel'] == rel)]['e1'].unique()
            tmp2.append([x for x in r.tolist() if x not in ent][:k])
        
        y_pred = torch.from_numpy(np.array(tmp2))
        #return (y_pred == y.reshape(-1,1)).any(1).float().mean()#.item()
        return torch.sum((y_pred == y.reshape(-1,1)).any(1).float())
    else:
        y_pred = torch.argsort(y_pred, dim=1, descending=True)[:,:k]
        #return (y_pred == y.reshape(-1,1)).any(1).float().mean()#.item()
        return torch.sum((y_pred == y.reshape(-1,1)).any(1).float())



def get_mrr(y, y_pred): #Mean Receiprocal Rank --> Average of rank of next item in the session.
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    y_pred = torch.argsort(y_pred, dim=1, descending=True)
    y = y.view(-1, 1).expand_as(y_pred)
    hits = (y == y_pred).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    return torch.sum(rranks)


def get_mr(y, y_pred): #Mean Rank --> Average of rank of next item in the session.
    """
    Calculates the MR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    y_pred = torch.argsort(y_pred, dim=1, descending=True)
    y = y.view(-1, 1).expand_as(y_pred)
    hits = (y == y_pred).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    return torch.sum(ranks)

