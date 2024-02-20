import time
import numpy as np
from TraStrainer import tra_strainer, process_metrics, read_traces
from baseline import random_sampling, sifter_sampling, sieve_sampling, wt_sampling
from baseline import read_traces as read_traces2

data_file = './data/hipster/test/0822/'
metrics = process_metrics(data_file)
traces0 = read_traces(data_file)
origin_traces, processed_traces = read_traces2(data_file + 'trace/')

processed_traces = processed_traces[:2000]
traces = {}
original_traces = {}
for t in processed_traces:
    traces[t.traceID] = traces0[t.traceID]
    original_traces[t.traceID] = origin_traces[t.traceID]

for sampling_rate in [0.001, 0.01, 0.025, 0.05, 0.1]:
    a = time.time()
    tra_strainer_result = tra_strainer(traces, metrics, sampling_rate)
    b = time.time()
    random_result = random_sampling(original_traces, sampling_rate)
    c = time.time()
    sifter_result = sifter_sampling(original_traces, sampling_rate)
    d = time.time()
    sieve_result = sieve_sampling(processed_traces, sampling_rate)
    e = time.time()
    wt_result = wt_sampling(processed_traces, sampling_rate)
    f = time.time()
    print(f"sampling_rate:{sampling_rate}, tra_strainer:{round(b - a, 5)}s, random:{round(c - b, 5)}s, "
          f"sifter:{round(d - c, 5)}s, sieve:{round(e - d, 5)}s, wt:{round(f - e, 5)}s")

for metric_dimention in range(0, 11):
    key_num = int(round(len(list(metrics.keys())) * metric_dimention / 10))
    new_metric = {}
    cnt = 0
    for key in metrics.keys():
        if cnt == key_num:
            break
        new_metric[key] = metrics[key]
        cnt += 1
    a = time.time()
    tra_strainer_result = tra_strainer(traces, new_metric, 0.001)
    b = time.time()
    print(f"Metric Dimention:{metric_dimention / 10} Time Used:{round(b - a, 2)}s")
