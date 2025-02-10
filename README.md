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

***Câu hỏi ở đây là:***
1. PRG sinh ra số lớn đến mức nào, nằm trong khoảng nào, làm sao biết được khoảng nó sẽ nằm do mỗi tham só mô hình sẽ có độ lớn khác nhau
2. Server gửi thông tin các client khác cho mỗi client(sau khi ping). Tức là chỉ gửi cho những thằng đang online thôi phải không?
3. Client random 2 số private với pair_private. Thì 2 số này là số nguyên có độ lớn như thế nào? Hay random lớn nhỏ bao nhiêu cũng được
4. Mỗi client gửi public cho trusted server, gửi thêm id của client. Thì id ở đây là id của chính client đang gửi phải không. Và id này có tác dụng gì?
5. g và p1 không đổi. Vậy server sẽ gửi tụi nó khi tụi nó đăng kí được không? Rồi server mới ping để kiểm tra coi thằng nào còn online để train. Nếu làm vậy thì cos ảnh hưởng gì không so với gửi g và p1 cho client sau khi ping?
