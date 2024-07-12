import os
import time

def count_files_in_directory(directory):
    try:
        return len(os.listdir(directory))
    except FileNotFoundError:
        return None

def monitor_directory(directory, interval=1, duration=60, output_file="file_count_log.txt"):
    start_time = time.time()
    end_time = start_time + duration
    
    with open(output_file, "w") as f:
        while time.time() < end_time:
            file_count = count_files_in_directory(directory)
            if file_count is not None:
                f.write(f"{time.ctime()}: {file_count} files\n")
                print(f"{time.ctime()}: {file_count} files")
            else:
                f.write(f"{time.ctime()}: Directory not found\n")
                print(f"{time.ctime()}: Directory not found")
            time.sleep(interval)

if __name__ == "__main__":
    directory_to_monitor = "/private/wangchen/instance_model/instance_model_data_test_data_generate_latency/pkl" 
    monitor_interval = 1  # 每秒检测一次
    monitor_duration = 120  # 总共监测60秒
    log_file = "file_count_log.txt"  # 记录结果的文件

    monitor_directory(directory_to_monitor, monitor_interval, monitor_duration, log_file)
