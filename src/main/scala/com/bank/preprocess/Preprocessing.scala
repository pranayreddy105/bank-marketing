package com.bank.preprocess

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object PreProcessor {

    def encode(data: DataFrame, categoricalFeatures: List[String], numericalFeatures: List[String], path:String): DataFrame = {


      val stringIndexers = new StringIndexer()
        .setInputCols(categoricalFeatures.toArray)
        .setOutputCols(categoricalFeatures.toArray map (name => s"${name}_indexed"))

      val encoder = new OneHotEncoder()
        .setInputCols(stringIndexers.getOutputCols)
        .setOutputCols(stringIndexers.getOutputCols map (name => s"${name}_onehotencoded"))


      val assembler = new VectorAssembler()
        .setInputCols(encoder.getOutputCols++numericalFeatures)
        .setOutputCol("features")

      val pipeline = new Pipeline()
        .setStages(Array(stringIndexers, encoder, assembler))


      val pipelineModel = pipeline.fit(data)

      val df = pipelineModel.transform(data)

      savePreProcessPipeline(pipelineModel, path)

      df
    }

    private def savePreProcessPipeline(model: PipelineModel, path: String): Unit = {
        var outputPath = path+"train//preprocessing-pipeline//"
        model.write.overwrite().save(outputPath)
    }
}
