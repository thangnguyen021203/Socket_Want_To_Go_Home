import torch
import copy

def aggregate(client_updates):
    """
    Tổng hợp các tham số mô hình từ các client bằng cách tính trung bình có trọng số.

    Args:
        client_updates: Danh sách các tuple (state_dict, num_data) từ các client.

    Returns:
        state_dict: Trạng thái mô hình đã tổng hợp.
    """
    # Lấy state_dict đầu tiên để làm khung lưu kết quả
    avg_state_dict = copy.deepcopy(client_updates[0][0])

    # Tổng số dữ liệu từ tất cả client
    total_samples = sum(num_data for _, num_data in client_updates)

    # Duyệt qua từng tham số trong state_dict
    for key in avg_state_dict.keys():
        avg_state_dict[key] = sum(
            (state_dict[key] * num_data) for state_dict, num_data in client_updates
        ) / total_samples  # Chia theo tổng số dữ liệu để lấy trung bình có trọng số

    return avg_state_dict
