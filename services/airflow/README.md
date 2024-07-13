### How to run

- Export environmental variables
```shell
export AIRFLOW_HOME=$PWD/services/airflow
export PYTHONPATH=$PWD/src
source venv/bin/activate
```

- Kill running airflow
```shell
sudo kill $(ps -ef | grep "airflow" | awk '{print $2}')
```

- Start Apache Airflow
```shell
airflow webserver --daemon --log-file services/airflow/logs/webserver.log
airflow scheduler  --daemon --log-file services/airflow/logs/scheduler.log
```