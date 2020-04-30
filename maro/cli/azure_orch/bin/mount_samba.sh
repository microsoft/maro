sudo mkdir /code_repo
sudo chmod -R 777 /code_repo
sudo mount -t cifs -o username=$ADMIN_USERNAME,password=maro_dist //$SAMBA_IP/sambashare /code_repo