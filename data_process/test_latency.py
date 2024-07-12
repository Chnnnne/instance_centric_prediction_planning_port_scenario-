import re
from datetime import datetime

def parse_log_line(line):
    """Parses a log line and returns the timestamp and file count."""
    match = re.match(r"(.*): (\d+) files", line)
    if match:
        timestamp_str, file_count = match.groups()
        timestamp = datetime.strptime(timestamp_str, "%a %b %d %H:%M:%S %Y")
        return timestamp, int(file_count)
    return None, None

def calculate_file_change_speed(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    previous_timestamp, previous_count = None, None
    speeds = []
    total_files_changed = 0
    total_time = 0

    for line in lines:
        timestamp, file_count = parse_log_line(line)
        if timestamp and previous_timestamp:
            time_diff = (timestamp - previous_timestamp).total_seconds()
            file_diff = file_count - previous_count
            speed = file_diff / time_diff
            speeds.append((timestamp, speed))

            total_files_changed += abs(file_diff)
            total_time += time_diff

        previous_timestamp, previous_count = timestamp, file_count

    average_speed = total_files_changed / total_time if total_time > 0 else 0
    return speeds, average_speed, total_time

def main():
    log_file = "/private/wangchen/instance_model/instance_model_data_test_data_generate_latency/file_count_log_origin.txt"  # 替换为你的日志文件名

    speeds, average_speed, total_time = calculate_file_change_speed(log_file)

    for timestamp, speed in speeds:
        print(f"{timestamp}: {speed:.2f} files/second")
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {average_speed:.2f} files/second")

if __name__ == "__main__":
    main()