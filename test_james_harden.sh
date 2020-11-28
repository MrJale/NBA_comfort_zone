#!/bin/bash
source env.sh
mvn clean package
/usr/local/hadoop/bin/hdfs dfs -rm -r /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /kmeans/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ./shot_logs.csv /kmeans/input/
/usr/local/spark/bin/spark-submit --class NBA_kmeans --master=spark://$SPARK_MASTER:7077 target/KMeans-1.0-SNAPSHOT.jar hdfs://$SPARK_MASTER:9000/kmeans/input/ "james harden"
mvn clean

