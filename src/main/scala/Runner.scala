import org.apache.spark.sql.SparkSession
import com.bank.datareader.DataReader
import com.bank.utils.Util.splitFeatures
import com.bank.train.ModelTraining
import com.bank.inference.Inference

object Runner {

  def main(args: Array[String]) {

    val stage = args(0)   // train or inference
    val inputPath = args(1) // input data path
    val savePath = args(2)  // model save or model load path

    // Get the SparkSession
    val spark = getSparkSession

    if (stage == "train") {
        // Load the data
        val sourceDF = DataReader.csvReader(inputPath, spark)
        println(sourceDF.show())

        // Target Column
        val targetColumn = "Class"

        // Split features into numerical and categorical
        val (numericalFeatures, categoricalFeatures) = splitFeatures(sourceDF, targetColumn)
        println("numericalFeatures:"+numericalFeatures)
        println("categoricalFeatures:"+categoricalFeatures)

        val df = ModelTraining.labelIndexer(sourceDF, savePath, targetColumn)
        println(df.show())

        // Train the models
        ModelTraining.train(df, categoricalFeatures, numericalFeatures, savePath)
    }
    else {
        println("inference stage: "+savePath)
        val sourceDF = DataReader.csvReader(inputPath, spark)
        println(sourceDF.show())
        println(sourceDF.count())
        Inference.infer(sourceDF, savePath)
    }

    spark.stop()

  }

  def getSparkSession: SparkSession = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    spark
  }

}
