import sys
import os
import json
import re

#configfile = 'vmconfig.json'
def rename(array):
    return str(abs(hash(array)))

def config_create(configfile,outfolder,standardfile):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    with open(standardfile, 'r') as f:
        standard_data = json.load(f)
        f.close()

    with open(configfile, 'r') as f:
        config_data = json.load(f)
        for vm in config_data['virtual_machines']:
            tmp_data = standard_data
            current_vm = vm['name']            
            with open(outfolder+'/'+current_vm+'.json', 'w') as outfile:  
                tmp_data['parameters']['networkInterfaceName']['value']=current_vm
                
                tmp_data['parameters']['location']['value']=config_data['location']
                tmp_data['parameters']['networkSecurityGroupName']['value']=current_vm+'-nsg'
                tmp_data['parameters']['virtualNetworkName']['value']=config_data['virtual_machine_resource_group']+'-vnet'
                tmp_data['parameters']['publicIpAddressName']['value']=current_vm+'-ip'
                tmp_data['parameters']['virtualMachineName']['value']=current_vm
                tmp_data['parameters']['virtualMachineRG']['value']=config_data['virtual_machine_resource_group']
                tmp_data['parameters']['virtualMachineSize']['value']=vm['size']
                tmp_data['parameters']['adminUsername']['value']=config_data['admin_username']
                tmp_data['parameters']['adminPublicKey']['value']=config_data['admin_public_key']
                tmp_data['parameters']['diagnosticsStorageAccountName']['value']='dist'+rename(current_vm+config_data['virtual_machine_resource_group'])
                tmp_data['parameters']['diagnosticsStorageAccountId']['value']='Microsoft.Storage/storageAccounts/dist'+rename(current_vm) + rename(current_vm+config_data['virtual_machine_resource_group'])
                json.dump(tmp_data, outfile, indent=4)