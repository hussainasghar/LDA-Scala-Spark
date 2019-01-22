import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, explode, udf}

object LDA_Markets_Name {

  def main(args: Array[String]): Unit = {

    val input = "src/main/resources/grams/RealDeal.csv"


    val spark = SparkSession.
      builder
      .master("local[*]")
      .appName("LDA-Topic-Modeling")
      .getOrCreate()


    val dataSet = spark.read.option("header","true").option("inferschema","true").csv(input)


    val nameFeature = dataSet.select("name").cache()

  /*  val regex1 = """[^\w\s\.\$]"""
    val regex2 = """\s\w{2}\s"""

    import spark.implicits._

    def remove_string: String => String = _.replaceAll(regex1, "").replaceAll(regex2, " ")
    def remove_string_udf = udf(remove_string)

    val cleanedFeature = feature.na.drop
      .withColumn("raw_text",remove_string_udf($"raw_text"))
*/


    // val tokenizer = new Tokenizer().setInputCol("description").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setPattern(" ")
      .setInputCol("name")
      .setOutputCol("words")

    val nGrams = new NGram().setN(3).setInputCol(regexTokenizer.getOutputCol)

    val stopwords =
      spark.read.textFile("src/main/resources/stopwords_github").collect() ++ Range('a', 'z').map(_.toChar.toString) ++ Array("")


    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(nGrams.getOutputCol)
      .setOutputCol("filtered")
      .setStopWords(stopwords)


    val vectorizer = new CountVectorizer()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")
      .setVocabSize(1000)

    val idf = new IDF().setInputCol("features")


    //NumTopics depends on Domain Knowledge
    val k = 8
    val iterations = 50

    val lda = new LDA()
      .setK(k)
      .setMaxIter(iterations)
      .setOptimizer("em")

    // Make a PipeLine above above stages
    val pipeline = new Pipeline()
      .setStages(Array(regexTokenizer,nGrams, stopWordsRemover, vectorizer, idf,lda))

    //Save PipeLine
    //   pipeline.write.overwrite().save("/tmp/ldaDemo/pipeline")

    // Fit finds the internal parameters of a model that will be used to transform data
    val pipelineModel = pipeline.fit(nameFeature)

    //  Transforming applies the parameters to data
    val transformedModel = pipelineModel.transform(nameFeature)

    //  print(pipelineModel.stages)

    val vectorizerModel = pipelineModel.stages(3).asInstanceOf[CountVectorizerModel]

    /*EMLDAOptimizer produces a DistributedLDAModel,
     which stores not only the inferred topics but also the full training corpus and
    topic distributions for each document in the training corpus.
    */
    val ldaModel = pipelineModel.stages(5).asInstanceOf[DistributedLDAModel]

    // Get vocab
    val vocabList = vectorizerModel.vocabulary
    // udf( UserDefinedFunctions )
    val termsIdx2Str = udf { (termIndices: Seq[Int]) => termIndices.map(idx => vocabList(idx)) }

    val maxTermsPerTopic = 10
    // Review Results of LDA model with Online Variational Bayes
    val topics = ldaModel.describeTopics(maxTermsPerTopic)
      .withColumn("terms", termsIdx2Str(col("termIndices")))

    val topicNames = topics.select("topic", "terms", "termWeights").show(false)

    val zipUDF = udf { (terms: Seq[String], probabilities: Seq[Double]) => terms.zip(probabilities) }

    //  explode function creates a new row for each element in the given array or map column (in a DataFrame)
    val topicsTmp = topics.withColumn("termWithProb", explode(zipUDF(col("terms"), col("termWeights"))))

    val termDF = topicsTmp.select(
      col("topic").as("topicId"),
      col("termWithProb._1").as("term"),
      col("termWithProb._2").as("probability"))

    termDF.show( numRows = k*maxTermsPerTopic , false)


  }
}
