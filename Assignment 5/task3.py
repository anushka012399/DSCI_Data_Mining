import random
import csv
from blackbox import BlackBox
import time
import sys

def reservoir_sampling(stream, reservoir_size):
    reservoir = []
    for i, user in enumerate(stream):
        if i < reservoir_size:
            reservoir.append(user)
        else:
            prob = reservoir_size / (i + 1)
            if random.random() < prob:
                replace_idx = random.randint(0, reservoir_size - 1)
                reservoir[replace_idx] = user
    return reservoir

if __name__ == "__main__":
    random.seed(553)
    start_time = time.time()
    bx = BlackBox()
    input_file = sys.argv[1]
    num_of_asks = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    result_file = sys.argv[4]
    reservoir_size = 100

    

    with open(result_file, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["seqnum", "0_id", "20_id", "40_id", "60_id", "80_id"])
        for i in range(num_of_asks):
            stream = bx.ask(input_file, stream_size)
            reservoir = reservoir_sampling(stream, reservoir_size)
            output = [((i + 1) * stream_size), reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]]
            csvwriter.writerow(output)
    end_time = time.time()
    print(end_time-start_time)
