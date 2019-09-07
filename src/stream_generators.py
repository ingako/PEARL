#!/usr/bin/env python3

from skmultiflow.data import SEAGenerator
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import ConceptDriftStream

def prepare_led_streams(noise_1 = 0.1, noise_2 = 0.1, func=0, alt_func=0):
    stream_1 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_1,
                                 has_noise=False,
                                 n_drift_features=func)

    stream_2 = LEDGeneratorDrift(random_state=0,
                                 noise_percentage=noise_2,
                                 has_noise=False,
                                 n_drift_features=alt_func)
    return stream_1, stream_2

def prepare_agrawal_streams(noise_1 = 0.05, noise_2 = 0.1, func=0, alt_func=0, random_state=0):
    stream_1 = AGRAWALGenerator(classification_function=func,
                                random_state=random_state,
                                balance_classes=False,
                                perturbation=noise_1)

    stream_2 = AGRAWALGenerator(classification_function=alt_func,
                                random_state=0,
                                balance_classes=False,
                                perturbation=noise_2)

    return stream_1, stream_2

def prepare_sea_streams(noise_1 = 0.05, noise_2 = 0.1, func=0, alt_func=0):
    stream_1 = SEAGenerator(classification_function=func,
                            random_state=0,
                            balance_classes=False,
                            noise_percentage=noise_1)

    stream_2 = SEAGenerator(classification_function=alt_func,
                            random_state=0,
                            balance_classes=False,
                            noise_percentage=noise_2)

    return stream_1, stream_2

def prepare_hyperplane_streams(noise_1 = 0.05, noise_2 = 0.1):
    # incremental drift
    stream_1 = HyperplaneGenerator(noise_percentage=noise_1,
                                   random_state=0,
                                   n_drift_features=10,
                                   mag_change=0.01,
                                   sigma_percentage=0.1)

    # subtle incremental drift
    stream_2 = HyperplaneGenerator(noise_percentage=noise_2,
                                   random_state=0,
                                   n_drift_features=10,
                                   mag_change=0.001,
                                   sigma_percentage=0.1)

    return stream_1, stream_2

def prepare_concept_drift_stream(stream_1, stream_2, drift_position, drift_width):
    stream = ConceptDriftStream(stream=stream_1,
                                drift_stream=stream_2,
                                random_state=0,
                                position=drift_position,
                                width=drift_width)

    # stream.prepare_for_use()
    return stream

class StreamObject():
    def __init__(self, streams):
        self.stable_period = 5000
        self.streams = streams
        self.stream_idx = 0
        self.count = 0
        self.cur_stream = streams[0]

    def next_sample(self):
        if self.count % self.stable_period == 0 and self.count != 0:
            self.stream_idx = (self.stream_idx + 1) % len(self.streams)
            self.cur_stream = self.streams[self.stream_idx]

        self.count += 1
        X, y = self.cur_stream.next_sample()
        return X, y

    def get_data_info(self):
        return self.cur_stream.get_data_info()

def prepare_data():
    # agrawal with 2 concepts
    stream_1, stream_2 = prepare_agrawal_streams(noise_1=0.05, noise_2=0.05, alt_func=4)
    drift_stream_1 = prepare_concept_drift_stream(stream_1=stream_1,
                                                stream_2=stream_2,
                                                drift_position=0,
                                                drift_width=1000)
    drift_stream_1.prepare_for_use()

    stream_3, stream_4 = prepare_agrawal_streams(noise_1=0.05, noise_2=0.05, alt_func=4, random_state=42)
    drift_stream_2 = prepare_concept_drift_stream(stream_1=stream_4,
                                                  stream_2=stream_3,
                                                  drift_position=0,
                                                  drift_width=1000)
    drift_stream_2.prepare_for_use()

    return StreamObject([drift_stream_1, drift_stream_2])

if __name__ == '__main__':
    stream = prepare_data()
    print(stream.get_data_info())
    with open("recur_agrawal.csv", "w") as out:
        for count in range(0, 200000):
            X, y = stream.next_sample()
            for row in X:
                features = ",".join(str(v) for v in row)
                out.write(f"{features},{str(y[0])}\n")
