# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from maro.cli.process.executor import ProcessExecutor


def start_job(deployment_path: str, **kwargs):
    executor = ProcessExecutor()
    executor.start_job(deployment_path=deployment_path)


def stop_job(job_name: str, **kwargs):
    executor = ProcessExecutor()
    executor.stop_job(job_name=job_name)


def delete_job(job_name: str, **kwargs):
    executor = ProcessExecutor()
    executor.delete_job(job_name=job_name)


def list_jobs(**kwargs):
    executor = ProcessExecutor()
    executor.list_job()


def get_job_logs(job_name: str, **kwargs):
    executor = ProcessExecutor()
    executor.get_job_logs(job_name=job_name)
