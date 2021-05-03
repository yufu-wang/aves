import torch

class Loop_subdivider():
    '''
    Pytorch implementaion: Loop subdivision
    We used this to produce the subdivided version of AVES (aves_high_res.pt)
    
    '''

    def __init__(self, device='cpu'):
        self.device = device
    
    def init_fn(self, meshes):
        self.bn = len(meshes)
        self.meshes = meshes
        
        # get ajacency matrix
        V = meshes.num_verts_per_mesh().sum()
        edges_packed = meshes.edges_packed()
        e0, e1 = edges_packed.unbind(1)
        idx01 = torch.stack([e0, e1], dim=1)
        idx10 = torch.stack([e1, e0], dim=1)
        idx = torch.cat([idx01, idx10], dim=0).t()

        ones = torch.ones(idx.shape[1], dtype=torch.float32).to(self.device)
        A = torch.sparse.FloatTensor(idx, ones, (V, V))
        self.deg = torch.sparse.sum(A, dim=1).to_dense().long()
        self.idx = self.sort_idx(idx)

        # get edges of default mesh
        self.eij = self.get_edges(meshes)
        
        
    def __call__(self, meshes, weights=None):
        self.init_fn(meshes)
        even_verts = self.step_1()
        odd_verts, odd_weights = self.step_2(weights)
        
        new_verts = torch.cat([even_verts, odd_verts])
        new_faces = self.compute_faces()
        
        if odd_weights is not None:
            new_weights = torch.cat([weights, odd_weights], dim=0)
        else:
            new_weights = None
        
        return new_verts, new_faces, new_weights
        
    
    def step_1(self):
        idx = self.idx
        deg = self.deg
        verts = self.meshes.verts_packed()
        nv = len(deg)

        beta = torch.zeros(nv)
        beta[deg>3] = 3 / (8*deg[deg>3].float())
        beta[deg==3] = 3 / 16
        beta[deg<3] = 1 / 8
        alpha = 1 - deg * beta

        beta_stack = torch.repeat_interleave(beta, deg, dim=0)

        vert2vert = torch.zeros([len(deg), len(deg)])
        vert2vert[idx[0], idx[1]] = beta_stack
        vert2vert[range(nv), range(nv)] = alpha
        
        even_verts = vert2vert @ verts
        return even_verts
        
        
    def step_2(self, weights=None):
        verts = self.meshes.verts_packed()
        faces = self.meshes.faces_packed()
        faces_edges = self.meshes.faces_packed_to_edges_packed()
        edges = self.meshes.edges_packed()


        unique, count = torch.unique(faces_edges, sorted=True, return_counts=True)
        interior = count>= 2 
        exterior = ~interior

        vert2edge_vert = torch.zeros([len(edges), len(verts)])
        vert2edge_vert[faces_edges.flatten(), faces.flatten()] += 1/8
        vert2edge_vert[faces_edges.flatten(), faces[:, [1,2,0]].flatten()] += 3/8
        
        vert2edge_vert[exterior] = 0
        vert2edge_vert[exterior, edges[exterior,0]] += 1/2
        vert2edge_vert[exterior, edges[exterior,1]] += 1/2
        
        odd_verts = vert2edge_vert @ verts
        
        if weights is not None:
            odd_weights = vert2edge_vert @ weights
        else:
            odd_weights = None
       
        return odd_verts, odd_weights
    
    
    def compute_faces(self):
        verts = self.meshes.verts_packed()
        faces = self.meshes.faces_packed()
        faces_edges = self.meshes.faces_packed_to_edges_packed()
        
        odd_faces = faces_edges + len(verts)
        faces_1 = torch.cat([faces[:,[0]], odd_faces[:,[2,1]]], dim=1)
        faces_2 = torch.cat([faces[:,[1]], odd_faces[:,[0,2]]], dim=1)
        faces_3 = torch.cat([faces[:,[2]], odd_faces[:,[1,0]]], dim=1)

        new_faces = torch.cat([faces_1, faces_2, faces_3, odd_faces], dim=0)
        
        return new_faces
    
    
    def get_edges(self, meshes):
        verts_packed = meshes.verts_packed()
        vi = torch.repeat_interleave(verts_packed, self.deg, dim=0)
        vj = verts_packed[self.idx[1]]
        eij = torch.stack([vi, vj])
        return eij
    
    
    def sort_idx(self, idx):
        _, order = (idx[0] + idx[1]*1e-6).sort()

        return idx[:, order]

