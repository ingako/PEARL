#!/usr/bin/env python3

from stream_generator import *

max_samples = 400001

generator = 'agrawal'

for seed in range(0, 10):
    stream = RecurrentDriftStream(generator=generator,
                                  has_noise=False,
                                  random_state=seed)
    stream.prepare_for_use()
    print(stream.get_data_info())

    output_filename = f'agrawal-3/agrawal-3-{seed}.csv'
    with open(output_filename, 'w') as out:
        for _ in range(max_samples):
            X, y = stream.next_sample()

            out.write(','.join(str(v) for v in X[0]))
            out.write(f',{y[0]}')
            out.write('\n')
