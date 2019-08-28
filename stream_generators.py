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

def prepare_agrawal_streams(noise_1 = 0.05, noise_2 = 0.1, func=0, alt_func=0):
    stream_1 = AGRAWALGenerator(classification_function=func,
                                random_state=0,
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

stream_1, stream_2 = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.05)
drift_stream = prepare_concept_drift_stream(stream_1=stream_1,
                                            stream_2=stream_2,
                                            drift_position=1500,
                                            drift_width=1)

stream_3, stream_4 = prepare_hyperplane_streams(noise_1=0.05, noise_2=0.05)

stream = prepare_concept_drift_stream(drift_stream, stream_3, 0 , 1)
stream.prepare_for_use()

with open('recur_hyperplane.csv', 'w') as out:
    max_samples = 20000
    for i in range(0, max_samples):
        X, y = stream.next_sample()
        features = ",".join(str(v) for v in X[0])
        out.write(f"{features},{str(y[0])}\n")
