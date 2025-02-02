import logging

def setup_logger(log_file="training.log"):
    """
    Thiết lập logger để ghi log ra file và hiển thị trên console.
    """
    logger = logging.getLogger("FedAvg")
    logger.setLevel(logging.INFO)

    # Ghi log vào file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Hiển thị log ra console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Định dạng log
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Thêm handler vào logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
