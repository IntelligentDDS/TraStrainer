import os
import csv
import copy
import time
import random
import numpy as np
import pandas as pd
from treelib import Tree
from collections import deque
from datetime import datetime, timedelta
from LTSF_Linear.run_longExp import Argument, metric_infer


def process_metrics(path):
    data_dict = {}
    folder_path = os.path.join(path, 'metric')
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    pod_name = row.get('PodName') if row.get('PodName') else row.get('ServiceName')
                    time = row.get('Time')[:19] if row.get('Time') else row.get('time')[:19]
                    for metric, value in row.items():
                        if 'Byte' in metric or 'P95' in metric or 'P99' in metric or 'Syscall' in metric:
                            continue
                        if metric not in ['PodName', 'ServiceName', 'Time', 'time', 'TimeStamp', 'timestamp']:
                            key = (pod_name.split('-')[0], metric)
                            if key not in data_dict:
                                data_dict[key] = []
                            value = 0.0 if not value else value
                            data_dict[key].append({'date': time, 'value': float(value)})
    return data_dict


def timestamp2datetime(timestamp):
    dt_object = datetime.fromtimestamp(int(timestamp))
    formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date


def future_datetime(date_string, minutes):
    date_obj = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    new_date_obj = date_obj + timedelta(minutes=minutes)
    new_date_string = new_date_obj.strftime('%Y-%m-%d %H:%M:%S')
    return new_date_string


def read_traces(path):
    traces = {}
    folder_path = os.path.join(path, 'trace')
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        row['status'] = 'success'
                        row['StartTime'] = timestamp2datetime(row['StartTimeUnixNano'][:10])
                        row['EndTime'] = timestamp2datetime(row['EndTimeUnixNano'][:10])
                        if row['TraceID'] in traces.keys():
                            traces[row['TraceID']].append(row)
                        else:
                            traces[row['TraceID']] = [row]
            except FileNotFoundError:
                pass
    return traces


def build_tree(node, parent_child, tree, spans):
    if node not in parent_child.keys():
        return
    for event_id, child in enumerate(parent_child[node]):
        if tree.contains(spans[child]['SpanID']):
            continue
        tree.create_node(tag=spans[child]['SpanID'], identifier=spans[child]['SpanID'], parent=node, data=spans[child])
        build_tree(spans[child]['SpanID'], parent_child, tree, spans)


def build_trace_tree(spans):
    spans = sorted(spans, key=lambda x: x['StartTimeUnixNano'])
    parent_child = {}
    node_info = {}
    for i, span in enumerate(spans):
        if span['ParentID'] in parent_child.keys():
            parent_child[span['ParentID']].append(i)
        else:
            parent_child[span['ParentID']] = [i]
        node_info[span['SpanID']] = i
    tree = Tree()
    if not tree.contains(spans[0]['SpanID']):
        tree.create_node(tag=spans[0]['SpanID'], identifier=spans[0]['SpanID'], data=spans[0])
    build_tree(spans[0]['SpanID'], parent_child, tree, spans)
    return tree


def process_trace(spans):
    resources = ['sql']
    data_dict = {}
    resource_dict = {}
    for row in spans:
        status = row['status']
        pod_name = row['PodName']
        pod_name = pod_name.split('-')[0]
        operation_name = row['OperationName']
        span_id = row['SpanID']
        duration = int(row['Duration'])
        if pod_name not in data_dict:
            data_dict[pod_name] = []
        data_dict[pod_name].append({'span_id': span_id, 'duration': duration, 'status': status})
        for resource in resources:
            if resource in operation_name:
                key = (pod_name, resource)
                if key not in resource_dict:
                    resource_dict[key] = []
                resource_dict[key].append({'span_id': span_id, 'duration': duration})
    tree = build_trace_tree(spans)
    return data_dict, resource_dict, tree


def get_seq_span(spans, tree):
    basic_features = []
    for span in spans:
        basic_features.append('-'.join([str(tree.depth(span['SpanID'])), span['PodName'], span['OperationName'],
                                        span['status'], str(int(int(span['Duration']) / 1e4))]))
        basic_features.sort()
    return basic_features


def compute_feature_values(data_dict, metrics):
    feature_values = {}
    for key in metrics.keys():
        pod = key[0]
        if pod not in data_dict.keys():
            feature_values[key] = 0
            continue
        span_data = data_dict[pod]
        num_spans = len(span_data)
        avg_duration = sum([entry['duration'] for entry in span_data]) / num_spans if num_spans else 0
        num_failures = sum(1 for entry in span_data if entry['status'] == 'fail')
        feature_value = num_spans * avg_duration * (1 + num_failures)
        feature_values[key] = feature_value
    return feature_values


def compute_jaccord_similarity(spanline_1, spanline_2):
    cp_spanline_1 = copy.deepcopy(spanline_1)
    cp_spanline_2 = copy.deepcopy(spanline_2)
    intersection_cnt = 0
    for sl1 in spanline_1:
        if sl1 in cp_spanline_2:
            intersection_cnt += 1
            cp_spanline_2.pop(cp_spanline_2.index(sl1))
            cp_spanline_1.pop(cp_spanline_1.index(sl1))
    union_cnt = intersection_cnt + len(cp_spanline_1) + len(cp_spanline_2)
    span_seq_similarity = intersection_cnt / union_cnt if union_cnt else 0
    return span_seq_similarity


def compute_metrics_weights(metrics, start_time, end_time):
    metrics_weights = {}
    for key, value in metrics.items():
        input = pd.DataFrame(value)
        predict_idx = (input['date'] >= start_time).idxmax()
        predict_len = len(input[(input['date'] >= start_time) & (input['date'] < future_datetime(end_time, 1))])
        predict_len = max(predict_len, 1)
        model_id = "_".join(key)
        if not os.path.exists(f'./checkpoints/{model_id}.pth'):
            metrics_weights[key] = 1
            continue
        args = Argument(input[predict_idx - 96:predict_idx + predict_len], model_id=model_id)
        preds, trues = metric_infer(args, model=None)
        metrics_weight = abs(float(((trues - preds) / preds).mean()))
        metrics_weights[key] = metrics_weight
    return metrics_weights


def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) - 1


def system_biased_filter(history_trace_metrics, trace_metric, metrics_weights):
    sampling_rate = 0
    count = 0
    n_sigma_bag = {}
    for key, weight in metrics_weights.items():
        value = trace_metric[key]
        if not history_trace_metrics:
            mean, std = 0, 0
        else:
            mean, std = np.mean(list(history_trace_metrics[key])), np.std(list(history_trace_metrics[key]))
        n_sigma_bag[key] = abs(value - mean) / (std + 1e-5)
        sampling_rate += weight * n_sigma_bag[key]
        count += weight
    sampling_rate = tanh(sampling_rate / count)
    return sampling_rate


def output_metrics(metrics):
    opt = {}
    for key, value in metrics.items():
        s_key = str(key)
        s_value = len(value)
        opt[s_key] = s_value
    return opt


def output_dict(d):
    opt = {}
    for key, value in d.items():
        s_key = str(key)
        s_value = str(value)
        opt[s_key] = s_value
    return opt


def init_history_trace_metrics(history_trace_metrics, metrics, window_size):
    for key in metrics.keys():
        history_trace_metrics[key] = deque(maxlen=window_size)


def diversity_biased_filter(history_trace_structures, trace_structure, diversity_window):
    clustered_history_traces = {}
    for e in history_trace_structures:
        element = '+'.join(e)
        if element in clustered_history_traces.keys():
            clustered_history_traces[element] += 1
        else:
            clustered_history_traces[element] = 1
    similarity = 0
    max_trace_structure = ''
    for history_trace_structure in history_trace_structures:
        cur_similarity = compute_jaccord_similarity(history_trace_structure, trace_structure)
        if cur_similarity > similarity:
            similarity = max(similarity, cur_similarity)
            max_trace_structure = '+'.join(history_trace_structure)
    count = clustered_history_traces.get(max_trace_structure, 0)
    similarity_rate = round(similarity * count, 2)
    if similarity_rate:
        similarity_rate = 1 / similarity_rate
        diversity_window.append(similarity_rate)
        return similarity_rate / sum(diversity_window)
    else:
        return 1.0


def judge(system_rate, diversity_rate, strict):
    system_random, diversity_random = random.random(), random.random()
    is_system_sample = True if system_random <= system_rate else False
    is_diversity_sample = True if diversity_random <= diversity_rate else False
    if strict:
        return is_system_sample and is_diversity_sample, system_random, diversity_random
    else:
        return is_system_sample or is_diversity_sample, system_random, diversity_random


def print_num(num):
    return f"{round(num, 2):.2f}"


def tra_strainer(traces, metrics, budget_sample_rate):
    window_size = int(1 / budget_sample_rate)
    warm_up_size = 10
    trace_vectors = []
    history_trace_metrics = {}
    init_history_trace_metrics(history_trace_metrics, metrics, window_size)
    history_trace_structures = deque(maxlen=window_size)
    diversity_window = deque(maxlen=window_size)
    cnt = 0
    cur_sample_cnt = 0
    sample_trace_ids = []
    time_used = []
    for trace_id, trace in traces.items():
        start_time = time.time()
        data_dict, resource_dict, tree = process_trace(trace)
        trace_structure = get_seq_span(trace, tree)
        trace_metric = compute_feature_values(data_dict, metrics)
        metrics_weights = compute_metrics_weights(metrics, trace[0]['StartTime'], trace[0]['EndTime'])
        if cnt >= warm_up_size:
            system_rate = system_biased_filter(history_trace_metrics, trace_metric, metrics_weights)
            diversity_rate = diversity_biased_filter(list(history_trace_structures), trace_structure, diversity_window)
            strict = True if (cur_sample_cnt / cnt > budget_sample_rate) else False
            is_sample, system_random, diversity_random = judge(system_rate, diversity_rate, strict)
            if is_sample:
                cur_sample_cnt += 1
                sample_trace_ids.append(trace_id)
            print(f"TraceID:{trace_id}\t SystemRate:{print_num(system_rate)}/{print_num(system_random)}\t "
                  f"DiversityRate:{print_num(diversity_rate)}/{print_num(diversity_random)}\t IsAnd:"
                  f"{1 if strict else 0}\t Sample:{is_sample}\t CurSampleRate:{print_num(cur_sample_cnt / cnt)}")
        history_trace_structures.append(trace_structure)
        trace_vectors.append(output_dict(trace_metric))
        for key in metrics.keys():
            history_trace_metrics[key].append(trace_metric[key])
        cnt += 1
        end_time = time.time()
        time_used.append(end_time - start_time)
        if cnt >= int(round(budget_sample_rate * len(traces))):
            break
    return sample_trace_ids
