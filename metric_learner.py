import os
import pandas as pd
from datetime import datetime
from LTSF_Linear.utils.metrics import metric
from LTSF_Linear.run_longExp import Argument, metric_learner, metric_infer

folder_path = './data/train/'
script_name = "./LTSF_Linear/run_longExp.py"
file_names = os.listdir(folder_path)

for i, file_name in enumerate(file_names):
    print(f'[{str(datetime.now())[:19]}] {i}/{len(file_names)} {file_name}')
    input = pd.read_csv(os.path.join(folder_path, file_name))
    # Train
    args = Argument(input[:-4], model_id=file_name[:-4])
    model = metric_learner(args)
    # Test
    args = Argument(input[-100:], model_id=file_name[:-4])
    preds, trues = metric_infer(args, model)
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
