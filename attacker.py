import numpy as np
import scipy.sparse as sp
from torch.nn.modules.module import Module

import torch
from numba import jit
import numba


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

        # print(f'number of pertubations: {n_perturbations}')
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
class MyAttacker(BaseAttack):
    """
    Nettack class used for poisoning attacks on node classification models.
    """
    def __init__(self, model=None, nnodes=None, device='cpu'):
        super(MyAttacker, self).__init__(model, nnodes, device=device)


    def preprocess_graph(self, modified_adj):
        adj = modified_adj + sp.eye(modified_adj.shape[0])
        row_sum = adj.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5))
        adj_normalized = adj.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized
    

    def compute_logits(self, preprocessed_adj, features, target_node, W):
        """
        Compute the logits of the surrogate model, i.e. linearized GCN.

        Returns
        -------
        np.array, [N, K]
            The log probabilities for each node.
        """
        return preprocessed_adj.dot(preprocessed_adj).dot(features.dot(W))[target_node].toarray()[0]


    def strongest_wrong_class(self, logits_start, label_u, K):
        """
        Determine the incorrect class with largest logits.

        Parameters
        ----------
        logits: np.array, [N, K]
            The input logits
        label_u

        K

        Returns
        -------
        np.array, [N, L]
            The indices of the wrong labels with the highest attached log probabilities.
        """
        label_u_onehot = np.eye(K)[label_u]
        # print('label_u_onehot',type(label_u_onehot), label_u_onehot, label_u_onehot.shape)
        return (logits_start - 1000*label_u_onehot).argmax()


    def compute_alpha(self, n_start, S_d_start, d_min):
        """
        Approximate the alpha of a power law distribution.

        Parameters
        ----------
        n: int or np.array of int
            Number of entries that are larger than or equal to d_min

        S_d: float or np.array of float
            Sum of log degrees in the distribution that are larger than or equal to d_min
        
        d_min: int
            The minimum degree of nodes to consider

        Returns
        -------
        alpha: float
            The estimated alpha of the power law distribution
        """
        return n_start / (S_d_start - n_start * np.log(d_min - 0.5)) + 1


    def compute_log_likelihood(self, n_start, alpha_start, S_d_start, d_min):
        """
        Compute log likelihood of the powerlaw fit.
        
        Parameters
        ----------
        n: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        
        alpha: float
            The estimated alpha of the power law distribution
        
        S_d: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        
        d_min: int
            The minimum degree of nodes to consider
        
        Returns
        -------
        float: the estimated log likelihood
        """
        return n_start * np.log(alpha_start) + n_start * alpha_start * np.log(d_min) + (alpha_start + 1) * S_d_start


    def update_Sx(self, current_S_d, current_n, d_edges_old, d_edges_new, d_min):
        """
        Update on the sum of log degrees S_d and n based on degree distribution resulting from inserting or deleting a single edge.
        
        Parameters
        ----------
        current_S_d: float
            Sum of log degrees in the distribution that are larger than or equal to d_min.
        
        current_n: int
            Number of entries in the old distribution that are larger than or equal to d_min.
        
        d_edges_old: np.array, shape [N,] dtype int
            The old degree sequence.
        
        d_edges_new: np.array, shape [N,] dtype int
            The new degree sequence
        
        d_min: int
            The minimum degree of nodes to consider
        
        Returns
        -------
        new_S_d: float, the updated sum of log degrees in the distribution that are larger than or equal to d_min.
        
        new_n: int, the updated number of entries in the old distribution that are larger than or equal to d_min.
        """
        old_in_range = d_edges_old >= d_min
        new_in_range = d_edges_new >= d_min

        d_old_in_range = np.multiply(d_edges_old, old_in_range)
        d_new_in_range = np.multiply(d_edges_new, new_in_range)

        new_S_d = current_S_d - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
        new_n = current_n - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)
        return new_S_d, new_n


    def filter_chisquare(self, new_ratios, delta_cutoff):
        """
        if new_ratios < delta_cutoff 
        
        return TRUE
        """
        return new_ratios < delta_cutoff
    

    def compute_new_a_hat_uv(self, potential_edges, modified_adj, preprocessed_adj, target_node, N):
        """
        Compute the updated A_hat_square_uv entries that would result
        from inserting/deleting the input edges, for every edge.

        P is len(possible_edges).

        Parameters
        ----------
        potential_edges: np.array, shape [P,2], dtype int

        modified_adj

        preprocessed_adj

        target_node

        N

        Returns
        ----------
        sp.sparse_matrix: updated A_hat_square_u entries, a sparse PxN matrix
        """

        edges = np.array(modified_adj.nonzero()).T
        edges_set = {tuple(x) for x in edges}
        A_hat_sq = preprocessed_adj @ preprocessed_adj
        values_before = A_hat_sq[target_node].toarray()[0]
        node_ixs = np.unique(edges[:, 0], return_index=True)[1]
        twohop_ixs = np.array(A_hat_sq.nonzero()).T
        degrees = modified_adj.sum(0).A1 + 1

        # reflected list, edge_set, is scheduled for deprecation
        # sol: https://stackoverflow.com/questions/68123204/numba-how-to-avoid-type-reflected-list-found-for-argument-warning
        ixs, vals = compute_new_a_hat_uv(edges, node_ixs, numba.typed.List(edges_set), twohop_ixs, values_before, degrees, potential_edges, target_node)

        ixs_arr = np.array(ixs)
        a_hat_uv = sp.coo_matrix((vals, (ixs_arr[:, 0], ixs_arr[:, 1])), shape = [len(potential_edges), N])

        return a_hat_uv


    def compute_XW(self, features, W):
        """
        Shortcut to compute the dot product of features and Weights
        """
        return features.dot(W)


    def struct_score(self, a_hat_uv, XW, label_u):
        """
        Compute structure scores, cf. Eq. 15 in the paper.

        P is len(possible_edges).

        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge

        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.
        
        label_u

        Returns
        ----------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[label_u]

        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:, label_u]

        struct_scores = logits_for_correct_class - best_wrong_class_logits

        return struct_scores
    

    def check_adj(self, modified_adj):
        """
        Check if the modified adjacency is symmetric and unweighted.
        """
        if type(modified_adj) is torch.Tensor:
            modified_adj = modified_adj.cpu().numpy()
        
        assert np.abs(modified_adj - modified_adj.T).sum() == 0, "Input graph is not symmetric"


    def attack(self, ori_features: sp.csr_matrix, ori_adj: sp.csr_matrix, labels: np.ndarray,
               idx_train: np.ndarray, target_node: int, n_perturbations: int, W1, W2, **kwargs):
        """
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
        target_node(u) : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could be edge removals/additions.
        """
        # print(f'number of pertubations: {n_perturbations}')

        delta_cutoff = 0.004
        
        # Adjacency matrix
        adj = ori_adj.copy().tolil()
        modified_adj = ori_adj.tolil()

        adj_no_selfloops = ori_adj.copy()
        adj_no_selfloops.setdiag(0)
        N = modified_adj.shape[0]
        preprocessed_adj = self.preprocess_graph(modified_adj).tolil()


        # Features
        features = ori_features.copy().tolil()
        

        # Node labels
        label = labels.copy()
        label_u = label[target_node]
        K =  np.max(label) + 1


        # GCN weight matrices
        W = sp.csr_matrix(W1.dot(W2))


        # Setting
        structure_perturbations = []
        potential_edges = []

        logits_start = self.compute_logits(preprocessed_adj, features, target_node, W)
        best_wrong_class = self.strongest_wrong_class(logits_start, label_u, K)
        surrogate_losses = [logits_start[label_u] - logits_start[best_wrong_class]]
        # print('logits_start',type(logits_start), logits_start,logits_start.shape)
        # print('surrogate_losses', type(surrogate_losses), surrogate_losses)


        # Starting Attack
        # print("##### Starting attack #####")


        # print("##### Attack only using structure perturbations #####")
        # Setup starting values of the likelihood ratio test.
        degree_sequence_start = adj.sum(0).A1
        current_degree_sequence = adj.sum(0).A1

        d_min = 2

        S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
        current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
        
        n_start = np.sum(degree_sequence_start >= d_min)
        current_n = np.sum(current_degree_sequence >= d_min)
        
        alpha_start = self.compute_alpha(n_start, S_d_start, d_min)
        log_likelihood_orig = self.compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)
        # print('alpha_start', type(alpha_start), alpha_start)
        # print('log_likelihood_orig', type(log_likelihood_orig), log_likelihood_orig)

        # print("##### Attacking the node directly #####")
        potential_edges = np.column_stack((np.tile(target_node, N-1), np.setdiff1d(np.arange(N), target_node)))
        potential_edges = potential_edges.astype("int32")

        for _ in range(n_perturbations):

            # Do not consider edges that, if removed, result in singleton edges in the graph.
            # Update the values for the power law likelihood ratio test.
            deltas = 2 * (1 - adj[tuple(potential_edges.T)].toarray()[0] )- 1
            
            d_edges_old = current_degree_sequence[potential_edges]
            d_edges_new = current_degree_sequence[potential_edges] + deltas[:, None]
            
            new_S_d, new_n = self.update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
            new_alphas = self.compute_alpha(new_n, new_S_d, d_min)
            new_ll = self.compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)

            n_combined = new_n + n_start
            S_d_combined = new_S_d + S_d_start
            alphas_combined = self.compute_alpha(n_combined, S_d_combined, d_min)
            ll_combined = self.compute_log_likelihood(n_combined, alphas_combined, S_d_combined, d_min)
            
            new_ratios = -2 * ll_combined + 2 * (new_ll + log_likelihood_orig)

            # Do not consider edges that, if added/removed, would lead to a violation of the likelihood ration Chi_square cutoff value.
            powerlaw_filter = self.filter_chisquare(new_ratios, delta_cutoff)
            potential_edges_final = potential_edges[powerlaw_filter]


            # Compute new entries in A_hat_square_uv
            a_hat_uv_new = self.compute_new_a_hat_uv(potential_edges_final, adj, preprocessed_adj, target_node, N)
            # print('a_hat_uv_new', type(a_hat_uv_new), a_hat_uv_new)


            # Compute the struct scores for each potential edge
            XW = self.compute_XW(features, W)
            struct_scores = self.struct_score(a_hat_uv_new, XW, label_u)
            # print('struct_scores', type(struct_scores), struct_scores)

            best_edge_idx = struct_scores.argmin()
            best_edge_score = struct_scores.min()
            best_edge = potential_edges_final[best_edge_idx]
            # print("Edge perturbation: {}".format(best_edge))


            # perform edge perturbation
            adj[tuple(best_edge)] = adj[tuple(best_edge[::-1])] = 1 - adj[tuple(best_edge)]
            preprocessed_adj = self.preprocess_graph(adj)
            structure_perturbations.append(tuple(best_edge))
            surrogate_losses.append(best_edge_score)


            # Update likelihood ratio test values
            current_S_d = new_S_d[powerlaw_filter][best_edge_idx]
            current_n = new_n[powerlaw_filter][best_edge_idx]
            current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_idx]
    
        # print('structure_perturbations',structure_perturbations)
        # print('number_perturbations',len(structure_perturbations))

        for node1, node2 in structure_perturbations:
            modified_adj[node1, node2] = 1 - modified_adj[node1, node2]
            modified_adj[node2, node1] = 1 - modified_adj[node2, node1]
        
        # print(((init_adj.toarray() - modified_adj.toarray())**2).sum() / 2)
        self.modified_adj = modified_adj
    # print("done")


@jit(nopython=True)
def connected_after(u, v, connected_before, delta):
    if u == v:
        if delta == -1:
            return False
        else:
            return True
    else:
        return connected_before


@jit(nopython=True)
def compute_new_a_hat_uv(edge_ixs, node_nb_ixs, edges_set, twohop_ixs, values_before, degs, potential_edges, u):
    """
    Compute the new values [A_hat_square]_u for every potential edge, where u is the target node. C.f. Theorem 5.1
    equation 17.

    Parameters
    ----------
    edge_ixs: np.array, shape [E,2], where E is the number of edges in the graph.
        The indices of the nodes connected by the edges in the input graph.
    
    node_nb_ixs: np.array, shape [N,], dtype int
        For each node, this gives the first index of edges associated to this node in the edge array (edge_ixs).
        This will be used to quickly look up the neighbors of a node, since numba does not allow nested lists.
    
    edges_set: set((e0, e1))
        The set of edges in the input graph, i.e. e0 and e1 are two nodes connected by an edge
    
    twohop_ixs: np.array, shape [T, 2], where T is the number of edges in A_tilde^2
        The indices of nodes that are in the twohop neighborhood of each other, including self-loops.
    
    values_before: np.array, shape [N,], the values in [A_hat]^2_uv to be updated.
    
    degs: np.array, shape [N,], dtype int
        The degree of the nodes in the input graph.
    
    potential_edges: np.array, shape [P, 2], where P is the number of potential edges.
        The potential edges to be evaluated. For each of these potential edges, this function will compute the values
        in [A_hat]^2_uv that would result after inserting/removing this edge.
    
    u: int
        The target node
    
    Returns
    -------
    return_ixs: List of tuples
        The ixs in the [P, N] matrix of updated values that have changed
    
    return_values:
    """
    N = degs.shape[0]

    twohop_u = twohop_ixs[twohop_ixs[:, 0] == u, 1]
    nbs_u = edge_ixs[edge_ixs[:, 0] == u, 1]
    nbs_u_set = set(nbs_u)

    return_ixs = []
    return_values = []

    for ix in range(len(potential_edges)):
        edge = potential_edges[ix]
        edge_set = set(edge)
        degs_new = degs.copy()
        delta = -2 * ((edge[0], edge[1]) in edges_set) + 1
        degs_new[edge] += delta

        nbs_edge0 = edge_ixs[edge_ixs[:, 0] == edge[0], 1]
        nbs_edge1 = edge_ixs[edge_ixs[:, 0] == edge[1], 1]

        affected_nodes = set(np.concatenate((twohop_u, nbs_edge0, nbs_edge1)))
        affected_nodes = affected_nodes.union(edge_set)
        a_um = edge[0] in nbs_u_set
        a_un = edge[1] in nbs_u_set

        a_un_after = connected_after(u, edge[0], a_un, delta)
        a_um_after = connected_after(u, edge[1], a_um, delta)

        for v in affected_nodes:
            a_uv_before = v in nbs_u_set
            a_uv_before_sl = a_uv_before or v == u

            if v in edge_set and u in edge_set and u != v:
                if delta == -1:
                    a_uv_after = False
                else:
                    a_uv_after = True
            else:
                a_uv_after = a_uv_before
            a_uv_after_sl = a_uv_after or v == u

            from_ix = node_nb_ixs[v]
            to_ix = node_nb_ixs[v + 1] if v < N - 1 else len(edge_ixs)
            node_nbs = edge_ixs[from_ix:to_ix, 1]
            node_nbs_set = set(node_nbs)
            a_vm_before = edge[0] in node_nbs_set

            a_vn_before = edge[1] in node_nbs_set
            a_vn_after = connected_after(v, edge[0], a_vn_before, delta)
            a_vm_after = connected_after(v, edge[1], a_vm_before, delta)

            mult_term = 1 / np.sqrt(degs_new[u] * degs_new[v])

            sum_term1 = np.sqrt(degs[u] * degs[v]) * values_before[v] - a_uv_before_sl / degs[u] - a_uv_before / \
                        degs[v]
            sum_term2 = a_uv_after / degs_new[v] + a_uv_after_sl / degs_new[u]
            sum_term3 = -((a_um and a_vm_before) / degs[edge[0]]) + (a_um_after and a_vm_after) / degs_new[edge[0]]
            sum_term4 = -((a_un and a_vn_before) / degs[edge[1]]) + (a_un_after and a_vn_after) / degs_new[edge[1]]
            new_val = mult_term * (sum_term1 + sum_term2 + sum_term3 + sum_term4)

            return_ixs.append((ix, v))
            return_values.append(new_val)

    return return_ixs, return_values
