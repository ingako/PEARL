#!/usr/bin/env python3

import sys

path = r'../'
if path not in sys.path:
    sys.path.append(path)

from src.stream_generator import RecurrentDriftStream

max_samples = 400001

generator = 'agrawal'
concepts = [4, 0, 8, 1, 2, 6]

for seed in range(0, 10):
    stream = RecurrentDriftStream(generator=generator,
                                  concepts=concepts,
                                  has_noise=False,
                                  random_state=seed)
    stream.prepare_for_use()
    print(stream.get_data_info())

    output_filename = f'agrawal-6/agrawal-6-{seed}.csv'
    # output_filename = f'agrawal-6-temp/agrawal-6-temp-{seed}.csv'
    with open(output_filename, 'w') as out:
        for _ in range(max_samples):
            X, y = stream.next_sample()

            out.write(','.join(str(v) for v in X[0]))
            out.write(f',{y[0]}')
            out.write('\n')
