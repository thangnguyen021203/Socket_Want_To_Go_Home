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
    print("Starting modify_state_dicts")
    print("neighbors:", neighbors)

    for key in s:
        print(f"Processing key: {key}")
        for client_id in neighbors.keys():
            # print(f"Checking client_id: {client_id}")

            if client_id not in neighbors:
                # print(f"Error: client_id {client_id} not found in neighbors")
                continue

            try:
                _, _, _, pair_PRG = neighbors[client_id]
                # print(f"pair_PRG for {client_id}: {pair_PRG}")
            except ValueError as e:
                # print(f"Error unpacking neighbors[{client_id}]: {e}")
                continue

            try:
                if int(own_id) < int(client_id):
                    s[key] += pair_PRG
                    # print("Updated s[key] with +pair_PRG")
                else:
                    s[key] -= pair_PRG
                    # print("Updated s[key] with -pair_PRG")
            except Exception as e:
                print(f"Error in comparison or update: {e}")

        try:
            s[key] += self_PRG
            print("Final update with self_PRG")
        except Exception as e:
            print(f"Error updating with self_PRG: {e}")
