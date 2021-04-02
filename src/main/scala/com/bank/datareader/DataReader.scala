package com.bank.datareader

import org.apache.spark.sql.{DataFrame, SparkSession}

object DataReader {

    def csvReader(path: String, spark: SparkSession): DataFrame = {

        spark.read.format("csv").option("header","true").option("inferSchema","true").load(path)

    }

    def parquetReader(path: String, spark: SparkSession): DataFrame = {

      spark.read.format("parquet").load(path)

    }
}
