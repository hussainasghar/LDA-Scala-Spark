
name := "TopicModeling"

version := "1.0"

scalaVersion := "2.11.12"


libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.0"



// https://mvnrepository.com/artifact/com.datastax.spark/spark-cassandra-connector
libraryDependencies += "com.datastax.spark" %% "spark-cassandra-connector" % "2.3.0"

   /******** Optional Scala Parser******/
// https://mvnrepository.com/artifact/com.github.scopt/scopt
libraryDependencies += "com.github.scopt" %% "scopt" % "3.2.0"


/**************** SparkNLP **************/
//libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.5.0"

//libraryDependencies += "com.johnsnowlabs.nlp" %% "spark-nlp" % "1.6.3"