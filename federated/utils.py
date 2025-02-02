# import torch

# def average_weights(client_weights):
#     """
#     Tính trung bình các trọng số từ các client.
#     """
#     avg_weights = {}
#     for key in client_weights[0].keys():
#         avg_weights[key] = torch.mean(
#             torch.stack([client_weights[i][key] for i in range(len(client_weights))]), dim=0
#         )
#     return avg_weights
