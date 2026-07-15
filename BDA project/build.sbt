name := "HatefulMemesFCM"

version := "0.1"

scalaVersion := "2.12.17"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.4.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.4.1" % "provided"
)
