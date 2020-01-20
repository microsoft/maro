import os
import openpyxl
import re

LEVEL_DESCRIPTION = [\
    'fixed',
    'limited vessel capacity',
    'different vessel capacity',
    'sine orders',
    'order noise',
    'buffer ticks noise',
    'vessel speed & duration noise',
    'different vessel speed',
    '2 sine orders',
    ]


def read_result(log_path):
    random_shortage, online_shortage, solution_shortage, runner_shortage, runner_booking = -1, -1, -1, -1, -1
    try:
        solution = open(os.path.join(log_path, 'lp_solution.txt'), 'r')
        solution_shortage_line = solution.readlines()[-1]
        solution_shortage = int(re.findall(r'(\d+)', solution_shortage_line)[0])
    except:
        print(f'Error pasring solution: {log_path}')

    runner = open(os.path.join(log_path, 'runner.txt'), 'r')
    for line in runner.readlines():
        if '[Total]' not in line:
            continue
        runner_nums = re.findall(r'(\d+)', line)
        shortage = int(runner_nums[0])
        runner_booking = int(runner_nums[2])
        if random_shortage == -1:
            random_shortage = shortage
        elif online_shortage == -1:
            online_shortage = shortage
        else:
            runner_shortage = shortage
    
    return random_shortage, online_shortage, solution_shortage, runner_shortage, runner_booking

if __name__ == '__main__':
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "LP Test Result"
    sheet.append(['Topology', 'Level', 'Level Description', 'Total Ticks', \
        'Random Shortage', 'Online Shortage', 'Offline Solution', 'Offline Shortage', \
            'Offline Solution~Shortage Gap', 'Booking'])

    LOG_PATH = "/home/Jinyu/maro_release/tools/replay_lp/log/20200102/"
    SAVE_PATH = LOG_PATH + "online_offline_comparison.xlsx"

    for topology in ["4p_ssdd", "5p_ssddd", "6p_sssbdd", "22p_global_ratio"]:
        for tick in [224, 448, 1120]:
            for level in range(9):
                experiment_name = f'{topology}_lp_{tick}_l0.{level}'
                config_name = f'{topology}_l0.{level}'
                
                log_path = f'{LOG_PATH}{experiment_name}'
                try:
                    random_shortage, online_shortage, solution_shortage, runner_shortage, runner_booking = read_result(log_path)
                except:
                    print(f'Error reading {experiment_name}')
                    continue
                sheet.append([topology, f'l0.{level}', LEVEL_DESCRIPTION[level], tick, \
                    random_shortage, online_shortage, solution_shortage, runner_shortage, \
                        (runner_shortage - solution_shortage)/(runner_shortage if runner_shortage > 0 else 1), runner_booking])

            sheet.append([])

    workbook.save(SAVE_PATH)
