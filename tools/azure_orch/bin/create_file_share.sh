# storageAccountName=dist4222797289618520747
# resourceGroupName=maro_dist
# shareName=hahaha

storageAccountKey=$(az storage account keys list \
    --resource-group $resourceGroupName \
    --account-name $storageAccountName \
    --query "[0].value" | tr -d '"')

az storage share create \
    --account-name $storageAccountName \
    --account-key $storageAccountKey \
    --name $shareName \
    --quota 100 \
    --output none