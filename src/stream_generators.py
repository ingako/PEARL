#!/usr/bin/env python3

import numpy as np
from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.data import SineGenerator
from skmultiflow.data import STAGGERGenerator
from skmultiflow.data import LEDGeneratorDrift
from skmultiflow.data import MIXEDGenerator
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import ConceptDriftStream

class RecurrentDriftStream(ConceptDriftStream):
    def __init__(self, generator='agrawal', concepts=[0, 4]):
        super().__init__()
        self.stable_period = 10000
        self.streams = []
        self.cur_stream = None
        self.stream_idx = 0
        self.drift_stream_idx = 0
        self.count = 0
        self.n_feautres = 0
        self.generator = generator
        self.concepts = concepts
        self.random_state = 0
        self._random_state = check_random_state(self.random_state)
        self.width = 1000
        self.position = 4000

    def next_sample(self, batch_size=1):

        """ Returns the next `batch_size` samples.

        Parameters
        ----------
        batch_size: int
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(x))
            if self._random_state.rand() > probability_drift:
                X, y = self.cur_stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        self.count += 1
        if self.count % self.stable_period == 0 and self.count != 0:
            self.sample_idx = 0
            self.stream_idx = (self.stream_idx + 2) % len(self.streams)
            self.drift_stream_idx = (self.stream_idx + 1) % len(self.streams)

            print(f"swithing to concept {self.stream_idx} and {self.drift_stream_idx}")

            self.cur_stream = self.streams[self.stream_idx]
            self.drift_stream = self.streams[self.drift_stream_idx]

        return self.current_sample_x, self.current_sample_y.flatten()

    def get_data_info(self):
        return self.cur_stream.get_data_info()

    def prepare_for_use(self):
        for concept in self.concepts:
            if self.generator == 'agrawal':
                stream = AGRAWALGenerator(classification_function=concept,
                                          random_state=self.random_state,
                                          balance_classes=False,
                                          perturbation=0.05)
            elif self.generator == 'sea':
                stream = SEAGenerator(classification_function=concept,
                                      random_state=self.random_state,
                                      balance_classes=False,
                                      noise_percentage=0.05)
            elif self.generator == 'sine':
                stream = SineGenerator(classification_function=concept,
                                       random_state=self.random_state,
                                       balance_classes=False,
                                       has_noise=False)
            elif self.generator == 'stagger':
                stream = STAGGERGenerator(classification_function=concept,
                                          random_state=self.random_state,
                                          balance_classes = False)
            elif self.generator == 'mixed':
                stream = MIXEDGenerator(classification_function=concept,
                                        random_state=self.random_state,
                                        balance_classes = False)
            stream.prepare_for_use()
            self.streams.append(stream)

        self.cur_stream = self.streams[0]
        self.drift_stream = self.streams[1]
        self.n_features = self.cur_stream.n_features


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

if __name__ == '__main__':
    generator = 'stagger'
    stream = RecurrentDriftStream(generator)
    stream.prepare_for_use()
    print(stream.get_data_info())
    with open(f"recur_{generator}.csv", "w") as out:
        for count in range(0, 20000):
            X, y = stream.next_sample()
            for row in X:
                features = ",".join(str(v) for v in row)
                out.write(f"{features},{str(y[0])}\n")
