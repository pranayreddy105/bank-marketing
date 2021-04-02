package com.bank.train

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.DataFrame
import com.bank.preprocess.PreProcessor
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.CrossValidatorModel


object ModelTraining {

  private var lrScore: Double = 0.0
  private var rfScore: Double = 0.0
  private var gbtScore: Double = 0.0

  def labelIndexer(df: DataFrame, savePath: String, targetCol: String): DataFrame = {

    val labelEncoder = new StringIndexer().setInputCol(targetCol).setOutputCol(s"${targetCol}_indexed")
    val labelEncoderModel = labelEncoder.fit(df)
    val data = labelEncoderModel.transform(df)
    saveLabelEncoder(labelEncoderModel, savePath)

    data
  }

  private def saveLabelEncoder(model: StringIndexerModel, path: String): Unit = {
    var outputPath = path + "labelencoder//"
    model.write.overwrite().save(outputPath)
  }


  def train(data: DataFrame, categoricalFeatures: List[String], numericalFeatures: List[String], savePath: String): Unit = {
    val train = PreProcessor.encode(data, categoricalFeatures, numericalFeatures, savePath)

    val lrModel = trainLR(train, savePath)
    val rfModel = trainRF(train, savePath)
    val gbtModel = trainGBT(train, savePath)

    // Creating a map score to model
    val table = Map(lrScore -> lrModel,
      rfScore-> rfModel,
      gbtScore-> gbtModel)

    // Creating a list of scores
    val lstScores = List(lrScore, rfScore, gbtScore)

    val maxAreaUnderROCScore = lstScores.max

    println("maxAreaUnderROCScore: "+maxAreaUnderROCScore)

    val bestModel = table(lstScores.max)

    println("Saving the best model..")

    val path = savePath + "train//models//bestmodel//"

    bestModel.write.overwrite().save(path)

    println("Completed Saving the bestmodel to Disk...")


  }

  private def trainLR(data: DataFrame, savePath: String): CrossValidatorModel = {

      println("Logistic Regression Training Started..")
      val lr = new LogisticRegression().setMaxIter(5)
      lr.setFeaturesCol("features")
      lr.setLabelCol("Class_indexed")

      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .addGrid(lr.elasticNetParam, Array(0.3, 0.5))
        .build()

      val eval = new BinaryClassificationEvaluator()
      eval.setLabelCol("Class_indexed")


      val cv = new CrossValidator()
        .setEstimator(lr)
        .setEvaluator(eval)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5) // Use 3+ in practice
        .setParallelism(2)

      // Run cross-validation, and choose the best set of parameters.
      val cvLRModel = cv.fit(data)

      // Use the test set to measure the accuracy of the model on new data
      val train_predictions = cvLRModel.transform(data)

      // cvModel uses the best model found from the Cross Validation
      // Evaluate best model
      lrScore = eval.evaluate(train_predictions).toDouble

      println("AreaUnderROC Score for Training Data By Logistic Regression: " + lrScore)

      val path = savePath + "train//models//logisticregression//"

      println("Saving the Logistic Regression Model to Disk")

      //Save Model to Disk
      cvLRModel.write.overwrite().save(path)

      println("Completed Saving the Logistic Regression Model to Disk...")

      cvLRModel

  }

  private def trainRF(data: DataFrame, savePath: String): CrossValidatorModel = {

      println("Random Forest Training Started..")
      val rf = new RandomForestClassifier()
      rf.setFeaturesCol("features")
      rf.setLabelCol("Class_indexed")

      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.numTrees, Array(10, 20))
        .addGrid(rf.maxDepth, Array(2, 5, 9))
        .build()

      val eval = new BinaryClassificationEvaluator()
      eval.setLabelCol("Class_indexed")

      val cv = new CrossValidator()
        .setEstimator(rf)
        .setEvaluator(eval)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(5) // Use 3+ in practice
        .setParallelism(2)

      // Run cross-validation, and choose the best set of parameters.
      val cvLRModel = cv.fit(data)

      // Use the test set to measure the accuracy of the model on new data
      val train_predictions = cvLRModel.transform(data)

      // cvModel uses the best model found from the Cross Validation
      // Evaluate best model
      rfScore = eval.evaluate(train_predictions).toDouble

      println("AreaUnderROC Score for Training Data By Random Forest: " + rfScore)

      val path = savePath + "train//models//randomforest//"

      println("Saving the Random Forest Model to Disk")

      //Save Model to Disk
      cvLRModel.write.overwrite().save(path)

      println("Completed Saving the Random Forest Model to Disk...")

      cvLRModel
  }

  private def trainGBT(data: DataFrame, savePath: String): CrossValidatorModel = {

    println("GBTClassifier  Training Started..")
    val gbt = new GBTClassifier()
      gbt.setFeaturesCol("features")
      gbt.setLabelCol("Class_indexed")

    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxIter,Array(5,10))
      .addGrid(gbt.maxDepth,Array(2,3))
      .build()

    val eval = new BinaryClassificationEvaluator()
    eval.setLabelCol("Class_indexed")

    val cv = new CrossValidator()
      .setEstimator(gbt)
      .setEvaluator(eval)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5) // Use 3+ in practice
      .setParallelism(2)

    // Run cross-validation, and choose the best set of parameters.
    val cvLRModel = cv.fit(data)

    // Use the test set to measure the accuracy of the model on new data
    val train_predictions = cvLRModel.transform(data)

    // cvModel uses the best model found from the Cross Validation
    // Evaluate best model
    gbtScore = eval.evaluate(train_predictions).toDouble

    println("AreaUnderROC Score for Training Data By GBTClassifier: " + gbtScore)

    val path = savePath + "train//models//gbtclassifier//"

    println("Saving the GBTClassifier Model to Disk")

    //Save Model to Disk
    cvLRModel.write.overwrite().save(path)

    println("Completed Saving the GBTClassifier Model to Disk...")

    cvLRModel

  }
}