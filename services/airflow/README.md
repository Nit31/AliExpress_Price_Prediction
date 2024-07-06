### How to run

- Export environmental variables
```shell
export AIRFLOW_HOME=$PWD/services/airflow
export PYTHONPATH=$PWD/src
source venv/bin/activate
```

- Start Apache Airflow
```shell
sudo kill $(ps -ef | grep "airflow" | awk '{print $2}')
airflow webserver --daemon --log-file services/airflow/logs/webserver.log
airflow scheduler  --daemon --log-file services/airflow/logs/scheduler.log
```