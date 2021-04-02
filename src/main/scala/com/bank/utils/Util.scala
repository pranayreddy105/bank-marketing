package com.bank.utils

import org.apache.spark.sql.DataFrame

object Util {

  // Function to separate features into numerical and categorical
  def splitFeatures(df: DataFrame, target: String) = {
    val myList = df.dtypes
    var categoricalFeatures : List[String] = Nil
    var numericalFeatures : List[String] = Nil
    for ( item <- myList ) {
      val colName = item._1
      val colType = item._2
      if (colName != target) {
        if (colType == "StringType") {
          categoricalFeatures = categoricalFeatures :+ colName
        }
        else {
          numericalFeatures = numericalFeatures :+ colName
        }
      }
    }
    (numericalFeatures,categoricalFeatures)
  }
}
