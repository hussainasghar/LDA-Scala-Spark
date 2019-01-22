/*
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.{Tokenizer,Normalizer, _}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.{col, explode, udf}
import com.johnsnowlabs.nlp.RecursivePipeline


object TopicModelling_JhonSnow {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.
      builder
      .master("local[*]")
      .appName("LDA-Topic-Modeling")
      .getOrCreate()


    val dataSet = spark.read.option("header","true").option("inferschema","true")
      .csv("src/main/resources/grams/Alpha.csv")

    val descfeature = dataSet.select(col("description")).cache()

  /*  val regex1 = """[^\w\s\.\$]"""
    val regex2 = """\s\w{2}\s"""

    import spark.implicits._

    def remove_string: String => String = _.replaceAll(regex1, "").replaceAll(regex2, " ")
    def remove_string_udf = udf(remove_string)

    val cleanedFeature = feature.na.drop
      .withColumn("raw_text",remove_string_udf($"raw_text"))
*/

    val documentAssembler = new DocumentAssembler()
      .setInputCol("description")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setTargetPattern(" ")
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val stemmer = new Stemmer().setInputCols(Array("normalized")).setOutputCol("stem")


    val finisher = new Finisher().setInputCols(Array("stem")).setOutputCols("finish").setOutputAsArray(true)

    val stopwords =
      spark.read.textFile("src/main/resources/stopwords").collect() ++ Range('a', 'z').map(_.toChar.toString) ++ Array("")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("finish")
      .setOutputCol("filtered")
      .setStopWords(stopwords)

  //  val token_assembler = new TokenAssembler().setInputCols("filtered").setOutputCol("assembled")

    val vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(3000)
     // .setMinDF(2)
      .setMinTF(2)

    val idf = new IDF().setInputCol("features")//.setOutputCol("features")

    val numTopics = 5
    val iterations = 100

    val lda = new LDA()
      .setK(numTopics)
      .setMaxIter(iterations)
      .setOptimizer("em")


    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        normalizer,
        stemmer,
        finisher,
        stopWordsRemover,
        vectorizer,
        idf,
        lda
      ))

    val pipelineModel = pipeline.fit(descfeature)

    val transformedModel = pipelineModel.transform(descfeature).show(false)

    val vectorizerModel = pipelineModel.stages(6).asInstanceOf[CountVectorizerModel]


    val ldaModel = pipelineModel.stages(8).asInstanceOf[DistributedLDAModel]


    val topicsIndexes = ldaModel.describeTopics(maxTermsPerTopic = 8)
    println("The topics described by their top-weighted terms:")
    topicsIndexes.show(false)


    val vocabList = vectorizerModel.vocabulary

    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }


    val topics = ldaModel.describeTopics(maxTermsPerTopic = 8)
      .withColumn("terms", termsIdx2Str(col("termIndices")))

    val topicNames = topics.select("topic", "terms", "termWeights").show(false)

    val zipUDF = udf { (terms: Seq[String], probabilities: Seq[Double]) => terms.zip(probabilities) }

    val topicsTmp = topics.withColumn("termWithProb", explode(zipUDF(col("terms"), col("termWeights"))))

    val termDF = topicsTmp.select(
      col("topic").as("topicId"),
      col("termWithProb._1").as("term"),
      col("termWithProb._2").as("probability"))

    termDF.show(false)

    termDF.coalesce(1).write.option("header", "true").csv("src/main/resources/LDA_NLP")
  }
}

*/
