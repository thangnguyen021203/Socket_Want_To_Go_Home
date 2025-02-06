import time
import matplotlib.pyplot as plt
import torch
from config import CONFIG
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from utils.plot import plot_metrics

def aggregate(global_model, client_updates):
    """
    Tổng hợp các tham số mô hình từ các client bằng cách tính trung bình trọng số.
    
    Args:
        global_model: Mô hình toàn cục (torch.nn.Module).
        client_updates: Danh sách state_dict từ các client.
    
    Returns:
        state_dict: Trạng thái mô hình đã tổng hợp.
    """
    avg_state_dict = client_updates[0].copy()

    # Tính trung bình cho từng tham số trong mô hình
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.mean(
            torch.stack([client[key] for client in client_updates]), dim=0
        )
    
    # Trả về trạng thái mô hình đã tổng hợp
    return avg_state_dict 

    # Cập nhật mô hình toàn cục
    global_model.load_state_dict(avg_state_dict)

