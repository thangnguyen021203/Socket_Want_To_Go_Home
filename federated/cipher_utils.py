from Crypto.Util.number import getPrime
from Crypto.Cipher import AES
from Crypto.Util import Counter
import hashlib
import random

def generate_prime(bits):
    """
    Sinh số nguyên tố ngẫu nhiên với số bit cho trước.
    """
    return getPrime(bits)

def random_number(exponent=10):
    """
    Sinh số nguyên ngẫu nhiên từ 0 đến 2^exponent.
    """
    return random.randint(0, 2**exponent)

def aes_ctr_prg(seed: int, num_bytes: int = 8, scale_factor: float = 1e-8):
    """
    Sinh số giả ngẫu nhiên từ AES-CTR dưới dạng số thực nhỏ.

    - seed: Số nguyên đầu vào để tạo khóa.
    - num_bytes: Độ dài của số ngẫu nhiên (mặc định 8 byte = 64 bit).
    - scale_factor: Hệ số chia để đảm bảo số sinh ra nhỏ (mặc định 1e-3).
    """
    # Chuyển seed thành khóa AES (SHA-256 -> 16 byte)
    key = hashlib.sha256(str(seed).encode()).digest()[:16]

    # Dùng chính seed làm nonce (hoặc có thể sinh ngẫu nhiên)
    nonce = hashlib.md5(str(seed).encode()).digest()[:8]  # Nonce 64-bit

    # Tạo bộ đếm (Counter) 128-bit với nonce
    ctr = Counter.new(128, initial_value=int.from_bytes(nonce, "big"))

    # Tạo AES-CTR với khóa và bộ đếm
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    # Sinh số ngẫu nhiên dạng byte
    random_bytes = cipher.encrypt(b'\x00' * num_bytes)

    # Chuyển thành số thực nhỏ
    random_float = (int.from_bytes(random_bytes, "big") % (10**6)) * scale_factor

    return random_float

# a = aes_ctr_prg(5)
# print(a)