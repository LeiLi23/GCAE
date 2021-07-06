import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):  #将稀疏矩阵转换为元组形式(行列坐标，值，shape)
    if not sp.isspmatrix_coo(sparse_mx):  #判断sparse_mx是否coo_matrix
        sparse_mx = sparse_mx.tocoo()      #返回稀疏矩阵coo_matrix形式
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  #矩阵的行列索引
    values = sparse_mx.data   #矩阵的值
    shape = sparse_mx.shape   #矩阵的形状
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def preprocess_normalized(adj):
    #adj = sp.coo_matrix(adj)
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    #adj_normalized = np.ndarray(adj_normalized)
    return adj_normalized

def construct_feed_dict(adj_normalized, adj, features, placeholders): #构建输入字典并返回
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0 #对角线求和，确保对角元素和为0
    adj_triu = sp.triu(adj) #取出稀疏矩阵上三角部分的元素
    adj_tuple = sparse_to_tuple(adj_triu) #行列索引
    #print("adj_tuple", adj_tuple[0])
    edges = adj_tuple[0]#取出所有的边，里面没有重复，一对对关系，每个关系是用一组数表示
    edges_all = sparse_to_tuple(adj)[0]#所有的边，含对角线元素和上下对角矩阵的重复元素

    all_edge_idx = list(range(edges.shape[0])) #给所有的边一个编号，从0到n
    np.random.shuffle(all_edge_idx)#给所有的边打散
   # val_edge_idx = all_edge_idx[:num_val]#取二十分之一当作val data

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)#判断矩阵中所有元素是否都为true
        return np.any(rows_close) #判断矩阵中是否有一个为true

    edges_false = []#产生负样本的 edges
    l = len(edges)
    while len(edges_false) <l:
        idx_i = np.random.randint(0, 495)
        idx_j = np.random.randint(495, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):#如果随机出来的是正样本，跳过
            continue
        edges_false.append([idx_i, idx_j])
    return edges_all, edges, edges_false



def mask_test_edges_local_5FCV (adj, k):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0 #对角线求和，确保对角元素和为0
    adj_triu = sp.triu(adj) #取出稀疏矩阵上三角部分的元素
    adj_tuple = sparse_to_tuple(adj_triu)
    #print("adj_tuple", adj_tuple[0])
    edges = adj_tuple[0]#取出所有的边，里面没有重复，一对对关系，每个关系是用一组数表示
    edges_all = sparse_to_tuple(adj)[0]#所有的边，含对角线元素和上下对角矩阵的重复元素

    all_edge_idx = list(range(edges.shape[0])) #给所有的边一个编号，从0到n
    np.random.shuffle(all_edge_idx)#给所有的边打散

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)#判断矩阵中所有元素是否都为true
        return np.any(rows_close) #判断矩阵中是否有一个为true

    edges_false = []#产生负样本的 edges
    edges_pos=[]  #产生正样本的edges
    edges_neg = []   #产生负样本的edges
    for idx_i in range(495):  #
        idx_j = 494+k
        if idx_i == idx_j:
            continue
        elif ismember([idx_i, idx_j], edges_all):#如果随机出来的是正样本
            edges_pos.append([idx_i, idx_j])
    l = len(edges_pos)
    while len(edges_false) <l:
        idx_i = np.random.randint(0, 495)
        idx_j = np.random.randint(495, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):#如果随机出来的是正样本，跳过
            continue
        edges_false.append([idx_i, idx_j])
    print("the len of pos and neg", len(edges_pos), len(edges_neg))    #
    return edges_all, edges_pos, edges_false



def mask_test_edges_all_neg(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0 #对角线求和，确保对角元素和为0
    adj_triu = sp.triu(adj) #取出稀疏矩阵上三角部分的元素
    adj_tuple = sparse_to_tuple(adj_triu)
    #print("adj_tuple", adj_tuple[0])
    edges = adj_tuple[0]#取出所有的边，里面没有重复，一对对关系，每个关系是用一组数表示
    edges_all = sparse_to_tuple(adj)[0]#所有的边，含对角线元素和上下对角矩阵的重复元素

    all_edge_idx = list(range(edges.shape[0])) #给所有的边一个编号，从0到n
    np.random.shuffle(all_edge_idx)#给所有的边打散
   # val_edge_idx = all_edge_idx[:num_val]#取二十分之一当作val data

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)#判断矩阵中所有元素是否都为true
        return np.any(rows_close) #判断矩阵中是否有一个为true


    edges_false = []#产生负样本的 edges
    for i in range(0,495):
        for j in range(495,i+383):
             if ismember([i, j], edges_all):#如果随机出来的是正样本，跳过
                continue
             else:
                 edges_false.append([i, j])

    print("the length of false samples", len(edges_false))

    return edges_all, edges, edges_false
