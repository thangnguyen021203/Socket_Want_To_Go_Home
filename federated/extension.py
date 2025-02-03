import syft as sy

def encrypt_model(model, workers):
    """
    Mã hóa mô hình sử dụng PySyft Additive Secret Sharing.
    """
    return model.share(*workers)

def decrypt_model(model):
    """
    Giải mã mô hình sau khi tổng hợp.
    """
    return model.get()
