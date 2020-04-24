import sys
import os
import json
import re

#configfile = 'vmconfig.json'
def rename(array):
    return str(abs(hash(array)))


def configCreate(configfile,outfolder,standardfile):
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    # else:
    #     for root, dirs, files in os.walk(outfolder, topdown=False):
    #         for name in files:
    #             os.remove(os.path.join(root, name))
    #         for name in dirs:
    #             os.rmdir(os.path.join(root, name))
    #     os.rmdir(outfolder)
    #     os.mkdir(outfolder)

    with open(standardfile, 'r') as f:
        standard_data = json.load(f)
        f.close()

    with open(configfile, 'r') as f:
        config_data = json.load(f)
        for vm in config_data['virtualMachines']:
            tmp_data = standard_data
            current_vm = vm['name']            
            with open('./'+outfolder+'/'+current_vm+'.json', 'w') as outfile:  
                tmp_data['parameters']['networkInterfaceName']['value']=current_vm
                
                tmp_data['parameters']['location']['value']=config_data['location']
                tmp_data['parameters']['networkSecurityGroupName']['value']=current_vm+'-nsg'
                tmp_data['parameters']['virtualNetworkName']['value']=config_data['virtualMachineRG']+'-vnet'
                tmp_data['parameters']['publicIpAddressName']['value']=current_vm+'-ip'
                tmp_data['parameters']['virtualMachineName']['value']=current_vm
                tmp_data['parameters']['virtualMachineRG']['value']=config_data['virtualMachineRG']
                tmp_data['parameters']['virtualMachineSize']['value']=vm['size']
                tmp_data['parameters']['adminUsername']['value']=config_data['adminUsername']
                tmp_data['parameters']['adminPublicKey']['value']=config_data['adminPublicKey']
                tmp_data['parameters']['diagnosticsStorageAccountName']['value']='dist'+rename(current_vm+config_data['virtualMachineRG'])
                tmp_data['parameters']['diagnosticsStorageAccountId']['value']='Microsoft.Storage/storageAccounts/dist'+rename(current_vm) + rename(current_vm+config_data['virtualMachineRG'])
                json.dump(tmp_data, outfile, indent=4)


if __name__ == '__main__':
    params = sys.argv


    if len(params) == 1:
        config_file = raw_input("please input config file: ")
        folder_name = raw_input("please input output folder name: ")
        standard = raw_input("please input your base configuration file: ")
    elif len(params) != 4:
        print("{} {} {}".format('Usage: python', params[0], '<config-file> <output-folder> [base-config-file]'))
        sys.exit(0)
    else:
        config_file = params[1]
        folder_name = params[2]
        standard = params[3]
        configCreate(config_file, folder_name, standard)
        print('Generate virtual machine configuration successfully.')
