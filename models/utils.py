import torch

def compare_state_dicts(s1, s2):
    """
    So sánh tham số 2 mô hình.
    """
    if s1.keys() != s2.keys():
        return False  # Nếu có key khác nhau, state_dict khác nhau
    for key in s1:
        if not torch.equal(s1[key], s2[key]):  # So sánh từng tensor
            return False
    return True

def modify_state_dicts(s, own_id, neighbors, self_PRG):
    """
    Thay đổi giá trị tham số mô hình.
    """
    for key in s:
        for client_id in neighbors.keys():
            _, _, _, pair_PRG = neighbors[client_id]
            if own_id < client_id:
                s[key] += pair_PRG
            else:
                s[key] -= pair_PRG
        s[key] += self_PRG