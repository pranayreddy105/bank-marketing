package com.bank.inference

import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, StringIndexerModel}

object Inference {

  def infer(data: DataFrame,  path: String) : Unit = {

    val preprocessModelPath = path+"preprocessing-pipeline"
    val preprocessModel = PipelineModel.load(preprocessModelPath)

    val df = preprocessModel.transform(data)

    val modelPath = path+"models\\bestmodel"

    val model = CrossValidatorModel.load(modelPath)

    val predictions = model.transform(df)

    val labelEncoderPath = "C:\\Users\\Pranay.Bommineni\\IdeaProjects\\bank-marketing\\src\\main\\scala\\com\\bank\\data\\results\\labelencoder"

    val stringIndexerModel = StringIndexerModel.load(labelEncoderPath)

    val labels = stringIndexerModel.labelsArray

    val indexer = new IndexToString().setInputCol("prediction").setOutputCol("predicted_class").setLabels(labels.flatten)

    val final_predictions = indexer.transform(predictions)

    val result = final_predictions.select("Class", "prediction", "predicted_class")

    println(result.show())

//    result.repartition(1).write.format("csv").option("header","true").save()


  }
}
