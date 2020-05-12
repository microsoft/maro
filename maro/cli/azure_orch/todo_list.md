1. change the job config output path, modify all the path that relate to this
    utils.generate_job_config, line 116

2. change the rsync source directory in sync_code and dev_mode to avoid the influence of user project name, modify all the path relate to this
    utils.sync_code, utils.dev_mode

3. modify all the places using os.environ['PYTHONPATH']

4. the --include pattern of rsync command could be optimized, currently there are too many --include
    utils.pull_log, line 96

5. add path "/maro" on the god machine as another samba file share directories

6. upload updated resource_group_info.json to /maro/dist directories of god, maybe serveral places should be modified

7. change the scp target path to /maro/dist on god, modify all the path that relate to this
    provision.init_god, line 104

8. need to delete the node free resources info on redis when delete resource group node
    prob, line 42

9. support multiple GPU cards

10. make redis port as configurable

11. component path should be provided in meta config for each component, on longer need to be prompt in docker.launch_job

12. change the path in docker identical to the path outside docker
    docker.launch_job

13. launch a new prob job to watch the running status of thses jobs
    docker.launch_job

14. change the WORKDIR in docker file: "cpu.dist.df", as the WORKDIR is longer used when the above modifications has been finished

14. change all the names that is not propable, for example the admin_username on god, the paramenters.json file etc.. currently thses files has extra personal information


