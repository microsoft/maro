sudo apt install -y samba
echo -e "[sambashare]\n    comment = Samba on Ubuntu\n    path = /code_repo\n    read only = no\n    browsable = yes" | sudo tee -a /etc/samba/smb.conf
sudo service smbd restart
sudo ufw allow samba