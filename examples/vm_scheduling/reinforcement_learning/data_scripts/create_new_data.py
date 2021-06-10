import pandas as pd
import numpy as np


vm_table, vm, cpu = [], [], []

vmtable_data_path = "data/vmtable_10k.csv"
vm_table = pd.read_csv(vmtable_data_path, header=0, index_col=False)

vm_id = vm_table['vmid']
vm_subscription = vm_table['subscriptionid']
vm_deployment = vm_table['deploymentid']
vm_created = vm_table['vmcreated']
vm_lifetime = vm_table['lifetime']
vm_deleted = vm_table['vmdeleted']
vm_category = vm_table['vmcategory']
vm_core_total = vm_table['vmcorecountbucket']
vm_memory_total = vm_table['vmmemorybucket']

vm_cpu_data_path = "data/vm_cpu_readings-file-{}-of-195.csv"

for id in range(vm_id.shape[0]):
    if vm_created[id] > 0 and vm_created[id] <= 300 and vm_core_total[id] != 32:
        vm.append(vm_id[id])

        vm_table.append(
            [
                vm_id[id], vm_subscription[id], vm_deployment[id],
                vm_created[id], vm_lifetime[id], vm_deleted[id],
                vm_category[id], vm_core_total[id], vm_memory_total[id]
            ]
        )
        vm_table.append(
            [
                vm_id[id] + 100000, vm_subscription[id], vm_deployment[id],
                vm_created[id], vm_lifetime[id], vm_deleted[id],
                vm_category[id], vm_core_total[id], vm_memory_total[id]
            ]
        )


flag = False
for i in range(1, 196):
    if flag:
        break
    vm_table = pd.read_csv(vm_cpu_data_path.format(i), header=0, index_col=False)
    vm_time = vm_table['timestamp']
    vm_id = vm_table['vmid']
    vm_maxcpu = vm_table['maxcpu']
    for id in range(vm_id.shape[0]):
        if vm_id[id] in vm:
            cpu.append([vm_time[id] - 2, vm_id[id], vm_maxcpu[id]])
            cpu.append([vm_time[id] - 2, vm_id[id] + 100000, vm_maxcpu[id]])
        if vm_time[id] > 300:
            flag = True
            break

with open('vmtable.txt', 'a') as f:
    for _ in vm_table:
        f.writelines(str(_)[1:-1])
        f.writelines("\n")

with open('cpu.txt', 'a') as f:
    for _ in cpu:
        f.writelines(str(_)[1:-1])
        f.writelines("\n")
