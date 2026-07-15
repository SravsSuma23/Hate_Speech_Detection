import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.broadcast.Broadcast
import scala.io.Source
import java.net.URI
import org.apache.hadoop.fs.{FileSystem, Path}

object FastTextEmbeddingToHDFS {

  val EMBEDDING_DIM = 300
  val MAX_TOKENS = 80

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("FastText-Embedding-LSTM")
      .master("spark://spark-master:7077")
      .getOrCreate()

    import spark.implicits._

    // --------------------------------------------------
    // HDFS paths
    // --------------------------------------------------
    val RAW =
      "hdfs://namenode:8020/data/facebook_hateful_memes/processed_text_LSTM"

    val OUT =
      "hdfs://namenode:8020/data/facebook_hateful_memes/fasttext_lstm_embeddings"

    val FASTTEXT_PATH =
      "hdfs://namenode:8020/models/fasttext/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec"

    // --------------------------------------------------
    // STEP 1: Build vocabulary
    // --------------------------------------------------
    println("🔹 Extracting dataset vocabulary...")

    val vocab: Set[String] =
      spark.read.parquet(s"$RAW/train")
        .select(explode(col("text_tokens")).as("token"))
        .distinct()
        .as[String]
        .collect()
        .toSet

    println(s"✅ Vocabulary size: ${vocab.size}")

    // --------------------------------------------------
    // STEP 2: Load FastText vectors (vocab-pruned)
    // --------------------------------------------------
    println("🔹 Loading FastText vectors (vocab-pruned)...")

    val fs = FileSystem.get(
      new URI(FASTTEXT_PATH),
      spark.sparkContext.hadoopConfiguration
    )

    val stream = fs.open(new Path(FASTTEXT_PATH))

    val embeddings: Map[String, Array[Float]] =
      Source.fromInputStream(stream)
        .getLines()
        .filter(line => vocab.contains(line.takeWhile(_ != ' ')))
        .map { line =>
          val parts = line.split(" ")
          (parts.head, parts.tail.map(_.toFloat))
        }
        .toMap

    stream.close()

    println(s"✅ Loaded ${embeddings.size} FastText vectors")

    val bcEmbeddings: Broadcast[Map[String, Array[Float]]] =
      spark.sparkContext.broadcast(embeddings)

    // --------------------------------------------------
    // STEP 3: TYPED LSTM-READY SEQUENCE EMBEDDING
    // Output: Seq[Seq[Float]] → [80][300]
    // --------------------------------------------------
    val zeroVec: Seq[Float] = Seq.fill(EMBEDDING_DIM)(0.0f)

    val sequenceEmbedding =
      udf { tokens: Seq[String] =>
        val safeTokens = Option(tokens).getOrElse(Seq.empty)

        val seq =
          safeTokens.take(MAX_TOKENS).map { tok =>
            bcEmbeddings.value
              .getOrElse(tok, zeroVec.toArray)
              .toSeq
          }

        if (seq.length < MAX_TOKENS)
          seq ++ Seq.fill(MAX_TOKENS - seq.length)(zeroVec)
        else
          seq
      }

    // --------------------------------------------------
    // STEP 4: Process splits
    // --------------------------------------------------
    def processSplit(split: String): Unit = {
      spark.read.parquet(s"$RAW/$split")
        .withColumn("text_embedding_seq",
          sequenceEmbedding(col("text_tokens")))
        .select("id", "img", "text_embedding_seq", "label")
        .write
        .mode("overwrite")
        .parquet(s"$OUT/$split")

      println(s"✅ FastText LSTM embeddings saved for $split")
    }

    processSplit("train")
    processSplit("dev")
    processSplit("test")

    // --------------------------------------------------
    println("⏳ Keeping Spark UI alive for 5 minutes...")
    Thread.sleep(300000)

    spark.stop()
  }
}
