import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def concat_dirs():
    main_folder = './data/hipster/train'
    merged_data = {}
    subdirs = sorted(os.listdir(main_folder))
    for subdir in subdirs:
        subdir_path = os.path.join(main_folder, subdir)
        if '.DS_Store' in subdir_path:
            continue
        for filename in os.listdir(subdir_path):
            if filename.endswith('.csv'):
                csv_path = os.path.join(subdir_path, filename)
                file_name = os.path.splitext(filename)[0]
                file_name = file_name.split('-')[0]
                df = pd.read_csv(csv_path)
                if file_name in merged_data:
                    merged_data[file_name] = pd.concat([merged_data[file_name], df], ignore_index=True)
                else:
                    merged_data[file_name] = df
    for subdir in subdirs:
        subdir_path = os.path.join(main_folder, subdir)
        if '.DS_Store' in subdir_path:
            continue
        for filename in os.listdir(subdir_path):
            file_name = os.path.splitext(filename)[0]
            file_name = file_name.split('-')[0]
            merged_data[file_name].to_csv(f'./data/hipster/processed_train/{file_name}.csv', index=False)


def process_metric():
    csv_folder = './data/hipster/processed_train/'
    new_csv_folder = './data/hipster/processed_train/'

    if not os.path.exists(new_csv_folder):
        os.makedirs(new_csv_folder)

    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    def truncate_string(text):
        return text[:19]

    for csv_file in csv_files:
        if csv_file in ['dependency.csv', 'destination_50.csv', 'source_50.csv']:
            continue
        file_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(file_path)

        columns_to_keep = ['Time', 'TimeStamp'] 
        metric_columns = [col for col in df.columns if col not in columns_to_keep] 

        for metric_column in metric_columns:
            if metric_column in ['Time', 'TimeStamp', 'time', 'timestamp', 'PodName', 'ServiceName']:
                continue
            if 'Bytes' in metric_column or 'P95' in metric_column or 'P99' in metric_column \
                    or 'Syscall' in metric_column:
                continue

            if 'Time' not in df.columns:
                df.rename(columns={'time': 'Time'}, inplace=True)

            print(csv_file, metric_column)
            new_df = df[['Time', metric_column]]
            new_df.rename(columns={'Time': 'date'}, inplace=True)
            new_df['date'] = new_df['date'].apply(truncate_string)

            new_csv_file_name = f"{os.path.splitext(csv_file)[0]}_{metric_column}.csv"

            new_csv_file_path = os.path.join(new_csv_folder, new_csv_file_name)
            new_df.to_csv(new_csv_file_path, index=False)


concat_dirs()
process_metric()
