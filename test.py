import time
count = 0
while count<2:
    start_time = time.time()
    end_time = time.time()
    print(start_time)
    print("Waiting.....")
    while end_time - start_time < 5:
        end_time = time.time()
    print(end_time)
    print("Start collecting data again....")
    count+=1