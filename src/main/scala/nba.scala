import org.apache.spark.{SparkContext, SparkConf}
//import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
//import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.ml.clustering.KMeans
//import org.apache.spark.ml.evaluation.ClusteringEvaluator

object NBA_kmeans{

  case class nbaschema(Game_id:Integer,
                       matchup:String,
                       location:String,
                       w:String,
                       final_margin:Integer,
                       shot_number:Integer,
                       period:Integer,
                       game_clock:String,
                       shot_clock:Double,
                       dribbles:Integer,
                       touch_time:Double,
                       shot_dist:Double,
                       pts_type:Integer,
                       shot_result:String,
                       closest_defender:String,
                       closest_defender_player_id:Integer,
                       close_def_dist:Double,
                       fgm:Integer,
                       pts:Integer,
                       player_name:String,
                       player_id:Integer)

  def main (args: Array[String]){
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder.appName("NBA_kmeans").master("local[*]")getOrCreate()

    import spark.implicits._
    val df=spark.read
      .option("sep",",")
      .option("header","true")
      .option("inferSchema","true")
      .csv(args(0))
      .as[nbaschema]

    val df_player = df.filter($"player_name"===args(1))

    /*val assembler = new VectorAssembler().
      setInputCols(Array("shot_clock", "shot_dist", "close_def_dist")).
      setOutputCol("features")
    val traindata = assembler.transform(df_player)
      .select("features")*/

    /*val kmeans = new KMeans().setK(4).setSeed(1L)
    val model = kmeans.fit(df_player)
    val predictions = model.transform(df_player)*/

    val traindata = df_player.na.drop()
    val cols = Array("SHOT_CLOCK","SHOT_DIST","CLOSE_DEF_DIST")
    val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
    val featureDf = assembler.transform(traindata)

    val kmeans = new KMeans()
      .setK(4)
      .setFeaturesCol("features")
      .setPredictionCol("cluster")
    val kmeansModel = kmeans.fit(featureDf)
    println("The centers for four zones:")
    kmeansModel.clusterCenters.foreach(println)

    val predictDf = kmeansModel.transform(featureDf)

    val df1 = predictDf.groupBy("cluster").agg("SHOT_RESULT"->"count").orderBy("cluster").withColumnRenamed("count(SHOT_RESULT)","total")
    val df2 = predictDf.filter($"SHOT_RESULT"==="made").groupBy("cluster").agg("SHOT_RESULT"->"count").orderBy("cluster").withColumnRenamed("count(SHOT_RESULT)","made")
    //df1.show(10)
    //df2.show(10)
    val df3 = df1.join(df2,Seq("cluster")).orderBy(("cluster"))
    val df4 = df3.withColumn("hit_rate", col = df3("made")/df3("total"))
    //df4.show(10)
    val df5 = df4.orderBy(df4("hit_rate").desc)
    df5.show()

    val clu_list = df5.select("cluster").collect().map(_(0)).toList
    val hit_list = df5.select("hit_rate").collect().map(_(0)).toList

    val index:Int = clu_list(0).toString.toInt

    println("The center of the best zone: "+kmeansModel.clusterCenters(index))
    println("The hit rate in the best zone: "+hit_list(0))

    spark.stop()

  }


}
