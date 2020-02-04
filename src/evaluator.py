import copy
from collections import deque

import numpy as np
from sklearn.metrics import cohen_kappa_score

class Evaluator:

    @staticmethod
    def prequential_evaluation(classifier,
                               stream,
                               max_samples,
                               sample_freq,
                               metrics_logger):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        metrics_logger.info("count,accuracy,kappa,memory")

        for count in range(0, max_samples):
            X, y = stream.next_sample()

            # test
            prediction = classifier.predict(X, y)[0]

            window_actual_labels.append(y[0])
            window_predicted_labels.append(prediction)
            if prediction == y[0]:
                correct += 1

            classifier.handle_drift(count)

            if count % sample_freq == 0 and count != 0:

                accuracy = correct / sample_freq
                kappa = cohen_kappa_score(window_actual_labels, window_predicted_labels)
                memory_usage = classifier.get_size()

                print(f"{count},{accuracy},{kappa},{memory_usage}")
                metrics_logger.info(f"{count},{accuracy},{kappa},{memory_usage}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []

            # train
            classifier.partial_fit(X, y)

        print(f"length of candidate_trees: {len(classifier.candidate_trees)}")


    @staticmethod
    def prequential_evaluation_cpp(classifier,
                                   stream,
                                   max_samples,
                                   sample_freq,
                                   metrics_logger):
        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream);

        for count in range(0, max_samples):
            if not classifier.get_next_instance():
                break

            correct += 1 if classifier.process() else 0

            if count % sample_freq == 0 and count != 0:
                accuracy = correct / sample_freq
                candidate_tree_size = classifier.get_candidate_tree_group_size()
                tree_pool_size = classifier.get_tree_pool_size()

                print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                metrics_logger.info(f"{count},{accuracy}," \
                                    f"{candidate_tree_size},{tree_pool_size}")

                correct = 0
                window_actual_labels = []
                window_predicted_labels = []


    @staticmethod
    def prequential_evaluation_proactive_(classifier,
                                         stream,
                                         max_samples,
                                         sample_freq,
                                         metrics_logger):
        np.random.seed(0)

        import grpc
        import seqprediction_pb2
        import seqprediction_pb2_grpc

        from denstream.DenStream import DenStream

        clusterer = DenStream(lambd=0.1, eps=10, beta=0.5, mu=3)

        def fit_predict(clusterer, interval):
            x = [np.array([interval])]
            label = clusterer.fit_predict(x)[0]

            if label == -1:
                return interval

            return int(round(clusterer.p_micro_clusters[label].center()[0]))

        correct = 0
        window_actual_labels = []
        window_predicted_labels = []

        # proactive drift point prediction
        drift_interval_seq_len = 8
        next_adapt_state_points = deque()
        predicted_drift_points = deque()
        drift_interval_sequence = deque(maxlen=drift_interval_seq_len)
        last_actual_drift_point = 0
        num_request = 0

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream);

        with grpc.insecure_channel('localhost:50051') as channel:

            stub = seqprediction_pb2_grpc.PredictorStub(channel)

            for count in range(0, max_samples):
                if not classifier.get_next_instance():
                    break

                correct += 1 if classifier.process() else 0

                if classifier.drift_detected:
                    if classifier.is_state_graph_stable():
                        # predict the next drift point
                        temp = drift_interval_sequence.popleft()
                        response = stub.predict(
                                            seqprediction_pb2
                                                .SequenceMessage(seqId=count,
                                                                 seq=drift_interval_sequence)
                                        )
                        drift_interval_sequence.appendleft(temp)

                        # print(f"Predicted next drift point: {response.seq[0]}")
                        interval = fit_predict(clusterer, response.seq[0])
                        predicted_drift_points.append(interval)

                        drift_interval_sequence.append(interval)
                        last_actual_drift_point += interval

                    else:
                        # find actual drift point at num_instances_before
                        num_instances_before = classifier.find_last_actual_drift_point()

                        if num_instances_before > -1:
                            interval = count - num_instances_before - last_actual_drift_point
                            if interval < 0:
                                print("Failed to find the actual drift point")
                                # exit()
                            else:
                                interval = fit_predict(clusterer, interval)
                                drift_interval_sequence.append(interval)
                                last_actual_drift_point = count - num_instances_before

                                # train CPT
                                if len(drift_interval_sequence) >= drift_interval_seq_len:

                                    num_request += 1
                                    print(f"gRPC train request {num_request}: {drift_interval_sequence}")
                                    if stub.train(
                                                seqprediction_pb2
                                                    .SequenceMessage(seqId=num_request, seq=drift_interval_sequence)
                                            ).result:
                                        pass
                                    else:
                                        print("CPT training failed")
                                        exit()


                if classifier.is_state_graph_stable():
                    if len(predicted_drift_points) > 0:
                        predicted_drift_points[0] -= 1
                        if predicted_drift_points[0] <= 0:
                            predicted_drift_points.popleft()

                            if not classifier.drift_detected:
                                classifier.select_candidate_trees_proactively()

                                offset = 0
                                if len(next_adapt_state_points) > 0:
                                    offset = next_adapt_state_points[-1]
                                next_adapt_state_points.append(51 - offset)

                    if len(next_adapt_state_points) > 0:
                        next_adapt_state_points[0] -= 1
                        if next_adapt_state_points[0] <= 0:
                            next_adapt_state_points.popleft()

                            if not classifier.drift_detected:
                                classifier.adapt_state_proactively()
                else:
                    if len(predicted_drift_points) > 0:
                        predicted_drift_points = deque()
                    if len(next_adapt_state_points) > 0:
                        next_adapt_state_points = deque()


                if count % sample_freq == 0 and count != 0:
                    accuracy = correct / sample_freq
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                    print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                    metrics_logger.info(f"{count},{accuracy}," \
                                        f"{candidate_tree_size},{tree_pool_size}")

                    correct = 0
                    window_actual_labels = []
                    window_predicted_labels = []

    @staticmethod
    def prequential_evaluation_proactive(classifier,
                                         stream,
                                         max_samples,
                                         sample_freq,
                                         metrics_logger):
        np.random.seed(0)

        import grpc
        import seqprediction_pb2
        import seqprediction_pb2_grpc

        correct = 0

        # proactive drift point prediction
        drift_interval_seq_len = 8
        drift_interval_sequence = deque(maxlen=drift_interval_seq_len)
        last_actual_drift_point = 0
        num_request = 0

        metrics_logger.info("count,accuracy,candidate_tree_size,tree_pool_size")

        classifier.init_data_source(stream);

        with grpc.insecure_channel('localhost:50051') as channel:

            stub = seqprediction_pb2_grpc.PredictorStub(channel)

            for count in range(0, max_samples):
                if not classifier.get_next_instance():
                    break

                correct += 1 if classifier.process() else 0

                if classifier.drift_detected:
                    if classifier.is_state_graph_stable():
                        exit()

                # else:
                #     stub.train(seqprediction_pb2.SequenceMessage(seqId=num_request, seq=drift_interval_sequence))

                if count % sample_freq == 0 and count != 0:
                    accuracy = correct / sample_freq
                    candidate_tree_size = classifier.get_candidate_tree_group_size()
                    tree_pool_size = classifier.get_tree_pool_size()

                    print(f"{count},{accuracy},{candidate_tree_size},{tree_pool_size}")
                    metrics_logger.info(f"{count},{accuracy}," \
                                        f"{candidate_tree_size},{tree_pool_size}")

                    correct = 0
