## FedProx
Trong FedProx người ta sampling devices là uniformly\
set number device là 10\
Epoch là 1 và without systems heterogeneity

*Bài báo này chỉ plot*: 
- về số rounds với việc chọn ra số device tham gia lần lượt là 0 50 và 90%
- synthetic theo mức độ 0, 0.5 và 1

## zkFL
Chạy và plot về **Training Accuracy, Total training time, encryption time, aggregation time**
![alt text](/img/image.png)
![alt text](/img/image-1.png)

## zkP
![alt text](/img/image-2.png)

## A New Federated Learning Framework Against Gradient Inversion Attacks
![alt text](/img/image-3.png)
![alt text](/img/image4.png)
![alt text](/img/image5.png)
![alt text](/img/image6.png)

# Note
https://www.geeksforgeeks.org/how-to-generate-large-prime-numbers-for-rsa-algorithm/
![alt text](/img/random_prime.png)



# Task
Chọn model theo paper người khác.\
Có cần chạy với nhiều %client khác nhau và plot ra không?\
<!-- Coi lại việc huấn luyện, tại sao hội tụ bị dao động -->
Chiến lược chọn client như thế nào?\
tích hợp zkP với PairwiseMasking\
đôi khi bị deadlock khi Ping ->*đã ổn*\
Chưa plot thời gian tổng hợp, thời gian training\

Việc cần mở rộng mô hình:
- Cho server gửi thông tin các client khác (client_active bao gồm client_id, client_host, client_port) cho mỗi client (sau khi PING), truyền thêm tham số chung g, p1 (p1 là số nguyên tố) => dùng thuật toán check số giả nguyên tố
- Tạo thêm 1 trusted server
- Client:
    - Trước khi train:
        - Random 2 số private, pair_private
	    - Mỗi client gửi public = g^pair_private (mod p1) cho trusted server, gửi thêm id của client
	    - Trusted server gửi lại toàn bộ các public của các client khác cho mỗi client
	    - Client tính public^pair_private (mod p1) => chạy qua PRG => pair_PRG
	    - Client cho private chạy qua PRG => self_PRG
    - Sau khi train:
        - Biến số thực thành số nguyên (optional)
	    - Modify các tham số: tham số mới = tham số cũ + toàn bộ pair_PRG (với id của client < id của pair) - toàn bộ pair_PRG (với id của client > id của pair) + self_PRG


