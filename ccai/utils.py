import torch

def rotate_jac(jac, mat):
    """
    This function apply rotation transformation for those robots whose robot frame is not the world frame
    jac has shape: [..., operational space dim, joint dim]
    """
    while len(mat.shape) < len(jac.shape):
        # make sure both tensor have the same shape so that it doesn't broad case in a wrong way
        mat = mat.unsqueeze(0)
    jac_trans, jac_rot = torch.chunk(jac, chunks=2, dim=-2) # a
    jac_trans = mat @ jac_trans
    jac_rot = mat @ jac_rot
    return torch.cat((jac_trans, jac_rot), dim=-2)
