import psutil
import time
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def find_process_by_name(process_name):
    """
    根据进程名查找进程ID列表[2,5](@ref)
    """
    pids = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if process_name.lower() in proc.info['name'].lower():
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids

def monitor_process(pid, duration=60, interval=1):
    """
    监控指定进程的资源占用情况[1,3](@ref)
    """
    cpu_data = []
    memory_data = []
    timestamps = []
    
    try:
        process = psutil.Process(pid)
        process_name = process.name()
    except psutil.NoSuchProcess:
        print(f"错误: 进程ID {pid} 不存在")
        return None, None, None
    
    print(f"开始监控进程 {process_name} (PID: {pid})")
    print(f"持续时间: {duration}秒, 采样间隔: {interval}秒")
    print("-" * 50)
    
    start_time = time.time()
    samples = 0
    
    while time.time() - start_time < duration:
        try:
            # 获取CPU占用率[1,7](@ref)
            cpu_percent = process.cpu_percent(interval=interval)
            
            # 获取内存信息（返回MB单位）[2,7](@ref)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # RSS物理内存转换为MB
            
            cpu_data.append(cpu_percent)
            memory_data.append(memory_mb)
            timestamps.append(datetime.now().strftime("%H:%M:%S"))
            samples += 1
            
            print(f"时间: {timestamps[-1]} | CPU: {cpu_percent:6.2f}% | 内存: {memory_mb:8.2f} MB")
            
        except psutil.NoSuchProcess:
            print(f"错误: 进程 {pid} 在监控过程中退出")
            break
        except KeyboardInterrupt:
            print("\n监控被用户中断")
            break
    
    return cpu_data, memory_data, timestamps

def generate_statistics(cpu_data, memory_data, process_name, pid):
    """
    生成统计报告[3](@ref)
    """
    if not cpu_data or not memory_data:
        print("没有收集到数据")
        return
    
    # 创建DataFrame用于统计[3](@ref)
    df = pd.DataFrame({
        'CPU_Usage_%': cpu_data,
        'Memory_Usage_MB': memory_data
    })
    
    # 计算统计指标
    stats = {
        '指标': ['平均值', '最大值', '最小值', '中位数', '标准差', '样本数'],
        'CPU占用率 (%)': [
            np.mean(cpu_data),
            np.max(cpu_data),
            np.min(cpu_data),
            np.median(cpu_data),
            np.std(cpu_data),
            len(cpu_data)
        ],
        '内存占用 (MB)': [
            np.mean(memory_data),
            np.max(memory_data),
            np.min(memory_data),
            np.median(memory_data),
            np.std(memory_data),
            len(memory_data)
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    
    print("\n" + "="*60)
    print(f"进程 {process_name} (PID: {pid}) 资源占用统计报告")
    print("="*60)
    print(stats_df.to_string(index=False))
    
    # 显示峰值信息
    max_cpu_time = cpu_data.index(max(cpu_data))
    max_memory_time = memory_data.index(max(memory_data))
    
    print(f"\n峰值信息:")
    print(f"CPU最高占用: {max(cpu_data):.2f}% (样本 #{max_cpu_time + 1})")
    print(f"内存最高占用: {max(memory_data):.2f} MB (样本 #{max_memory_time + 1})")
    
    return stats_df

# python process_monitor.py --name "basalt_vio_ros_" --duration 120 --interval 1

def main():
    parser = argparse.ArgumentParser(description='进程资源监控工具')
    parser.add_argument('--name', type=str, help='进程名称（支持部分匹配）')
    parser.add_argument('--pid', type=int, help='进程ID')
    parser.add_argument('--duration', type=int, default=60, help='监控持续时间（秒）')
    parser.add_argument('--interval', type=float, default=1, help='采样间隔（秒）')
    
    args = parser.parse_args()
    
    if not args.name and not args.pid:
        print("请指定要监控的进程（使用 --name 或 --pid 参数）")
        return
    
    target_pid = args.pid
    
    # 如果通过进程名查找[5](@ref)
    if args.name and not args.pid:
        pids = find_process_by_name(args.name)
        if not pids:
            print(f"未找到包含 '{args.name}' 的进程")
            return
        elif len(pids) > 1:
            print(f"找到多个匹配进程: {pids}")
            print("请使用 --pid 参数指定具体的进程ID")
            return
        else:
            target_pid = pids[0]
    
    # 执行监控
    cpu_data, memory_data, timestamps = monitor_process(
        target_pid, args.duration, args.interval
    )
    
    if cpu_data and memory_data:
        # 生成统计报告
        process_name = psutil.Process(target_pid).name()
        stats = generate_statistics(cpu_data, memory_data, process_name, target_pid)
        
        # 可选：保存数据到CSV文件[2](@ref)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"process_monitor_{process_name}_{timestamp}.csv"
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'CPU_Usage_%': cpu_data,
            'Memory_Usage_MB': memory_data
        })
        df.to_csv(filename, index=False)
        print(f"\n详细数据已保存到: {filename}")

if __name__ == "__main__":
    main()