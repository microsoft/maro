# storageAccountName=dist4222797289618520747
# resourceGroupName=maro_dist
# shareName=hahaha

shareFileLocation="//${storageAccountName}.file.core.windows.net/${shareName}"
mountDir=/docker_images/
username=$shareName
password=az storage account keys list --resource-group maro_dist --account-name dist4222797289618520747 --query "[0].value" | tr -d '"'

auth="vers=3.0,username=${shareName},password=${password},dir_mode=0777,file_mode=0777,sec=ntlmssp"

sudo mount -t cifs $shareFileLocation $mountDir -o $auth
