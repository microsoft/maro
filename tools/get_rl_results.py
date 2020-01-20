import os
import openpyxl
import re
import csv

LEVEL_DESCRIPTION = {
    'A': 'fixed',
    'B': 'capacity',
    'C': 'sine with noise',
    'D': 'sine with noise + capacity',
    'E': 'speed[0.8, 1.0, 1.2]',
    'F': 'speed[0.8, 1.0, 1.2] + capacity',
    'G': 'speed[0.8, 1.0, 1.2] with noise',
    'H': 'speed[0.8, 1.0, 1.2] with noise + capacity',
    'I': 'sine with noise + speed[0.8, 1.0, 1.2] + capacity',
    'J': 'sine with noise + speed[0.8, 1.0, 1.2] with noise + capacity',
    'K': 'speed[0.8, 0.9, 1.0]',
    'L': 'speed[0.8, 0.9, 1.0] + capacity',
    'M': 'speed[0.8, 0.9, 1.0] with noise',
    'N': 'speed[0.8, 0.9, 1.0] with noise + capacity',
    'O': 'sine with noise + speed[0.8, 0.9, 1.0] + capacity',
    'P': 'sine with noise + speed[0.8, 0.9, 1.0] with noise + capacity',
}

def read_result(log_path):
    shortage, booking = -1, -1
    with open(os.path.join(log_path, 'runner.performance.csv'), 'r', encoding='utf-8') as performance:
        print('opened ' + log_path)
        reader = csv.reader(performance, delimiter=',')
        header = []
        for line in reader:
            if len(header) == 0:
                header = line
        shortage = line[header.index('total_shortage')]
        booking = line[header.index('total_booking')]
    return shortage, booking


if __name__ == '__main__':
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "DQN demo topology result"
    sheet.append(['Topology', 'Level', 'Level Description', 'Shortage', 'Booking'])

    LOG_PATH = "/home/jinywan/maro/examples/ecr/q_learning/single_host_mode/log/20200116/"
    SAVE_PATH = LOG_PATH + "dqn_demo_topology_456p.xlsx"

    for topology in ["4p_ssdd", "5p_ssddd", "6p_sssbdd"]:
        for level in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
            experiment_name = f'dqn_tc_{topology}_{level}'
            config_name = f'{topology}_{level}'

            log_path = f'{LOG_PATH}{experiment_name}'
            try:
                shortage, booking = read_result(log_path)
            except:
                print(f'Error reading {experiment_name}')
                continue
            sheet.append([topology, f'{level}', LEVEL_DESCRIPTION[level], shortage, booking])
        sheet.append([])

    workbook.save(SAVE_PATH)
