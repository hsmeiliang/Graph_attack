import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module


class BaseAttack(Module):
    """Abstract base class for target attack classes.
    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes, device='cpu'):
        super(BaseAttack, self).__init__()
        self.surrogate = model
        self.nnodes = nnodes
        self.device = device

        self.modified_adj = None

    def attack(self, ori_adj, n_perturbations, **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix.
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        Returns
        -------
        None.
        """
        raise NotImplementedError()


class RND(BaseAttack):
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(RND, self).__init__(model, nnodes, device=device)

    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, **kwargs):
        """
        Randomly sample nodes u whose label is different from v and
        add the edge u,v to the graph structure. This baseline only
        has access to true class labels in training set
        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """

        print(f'number of pertubations: {n_perturbations}')
        modified_adj = ori_adj.tolil()

        row = ori_adj[target_node].todense().A1
        diff_label_nodes = [x for x in idx_train if labels[x] != labels[target_node] and row[x] == 0]
        diff_label_nodes = np.random.permutation(diff_label_nodes)

        if len(diff_label_nodes) >= n_perturbations:
            changed_nodes = diff_label_nodes[: n_perturbations]
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
        else:
            changed_nodes = diff_label_nodes
            unlabeled_nodes = [x for x in range(ori_adj.shape[0]) if x not in idx_train and row[x] == 0]
            unlabeled_nodes = np.random.permutation(unlabeled_nodes)
            changed_nodes = np.concatenate([changed_nodes, unlabeled_nodes[: n_perturbations-len(diff_label_nodes)]])
            modified_adj[target_node, changed_nodes] = 1
            modified_adj[changed_nodes, target_node] = 1
            pass

        self.modified_adj = modified_adj


# TODO: Implemnet your own attacker here
# class MyAttacker(BaseAttack):
#     def __init__(self, model=None, nnodes=None, device='cpu'):
#         super(MyAttacker, self).__init__(model, nnodes, device=device)

#     def attack(self, ori_features, ori_adj, target_node, n_perturbations, **kwargs):
#         pass
