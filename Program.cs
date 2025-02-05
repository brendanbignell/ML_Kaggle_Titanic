using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using System.Text.Json;

public class Program
{
    // Define the data structure for training data
    public class TitanicTrainData
    {
        [LoadColumn(0)]
        public float PassengerId { get; set; }

        [LoadColumn(1)]
        public bool Survived { get; set; }

        [LoadColumn(2)]
        public float Pclass { get; set; }

        [LoadColumn(3)]
        public string? Name { get; set; }

        [LoadColumn(4)]
        public string? Sex { get; set; }

        [LoadColumn(5)]
        public float Age { get; set; }

        [LoadColumn(6)]
        public float SibSp { get; set; }

        [LoadColumn(7)]
        public float Parch { get; set; }

        [LoadColumn(8)]
        public string? Ticket { get; set; }

        [LoadColumn(9)]
        public float Fare { get; set; }

        [LoadColumn(10)]
        public string? Cabin { get; set; }

        [LoadColumn(11)]
        public string? Embarked { get; set; }
    }

    // Define the data structure for test data
    public class TitanicTestData
    {
        [LoadColumn(0)]
        public float PassengerId { get; set; }

        [LoadColumn(1)]
        public float Pclass { get; set; }

        [LoadColumn(2)]
        public string? Name { get; set; }

        [LoadColumn(3)]
        public string? Sex { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        [LoadColumn(5)]
        public float SibSp { get; set; }

        [LoadColumn(6)]
        public float Parch { get; set; }

        [LoadColumn(7)]
        public string? Ticket { get; set; }

        [LoadColumn(8)]
        public float Fare { get; set; }

        [LoadColumn(9)]
        public string? Cabin { get; set; }

        [LoadColumn(10)]
        public string? Embarked { get; set; }
    }

    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }
    }


    static void Main(string[] args)
    {
        // Check CUDA availability
        try
        {
            Console.WriteLine("Checking CUDA availability...");
            var cudaEnv = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (string.IsNullOrEmpty(cudaEnv))
            {
                Console.WriteLine("Warning: CUDA_PATH environment variable not found. GPU acceleration might not be available.");
            }
            else
            {
                Console.WriteLine($"CUDA found at: {cudaEnv}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Error checking CUDA availability: {ex.Message}");
        }

        var mlContext = new MLContext(seed: 0);

        // Load data
        Console.WriteLine("Loading data...");
        var trainingDataView = mlContext.Data.LoadFromTextFile<TitanicTrainData>(
            path: "../../../train.csv",
            hasHeader: true,
            separatorChar: ',');

        var testDataView = mlContext.Data.LoadFromTextFile<TitanicTestData>(
            path: "../../../test.csv",
            hasHeader: true,
            separatorChar: ',');

        // Create data processing pipeline
        var dataPipeline = mlContext.Transforms
            .Categorical.OneHotEncoding(outputColumnName: "SexEncoded", inputColumnName: "Sex")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EmbarkedEncoded", inputColumnName: "Embarked"))
            .Append(mlContext.Transforms.ReplaceMissingValues(
                new[] {
                    new InputOutputColumnPair("Age", "Age"),
                    new InputOutputColumnPair("Fare", "Fare")
                }))
            .Append(mlContext.Transforms.Concatenate("Features",
                "Pclass", "SexEncoded", "Age", "SibSp", "Parch", "Fare", "EmbarkedEncoded"));

        var results = new Dictionary<string, object>();

        // Train both models
        Console.WriteLine("\nTraining Standard LightGBM with GPU...");
        var standardLgbm = TrainStandardLightGBM(mlContext, trainingDataView, testDataView, dataPipeline);
        results["Standard LightGBM"] = standardLgbm;

        Console.WriteLine("\nTraining FastTree with GPU...");
        var fastTreeLgbm = TrainFastTreeLightGBM(mlContext, trainingDataView, testDataView, dataPipeline);
        results["FastTree"] = fastTreeLgbm;

        PrintResults(results);
    }

    static ModelResults TrainStandardLightGBM(MLContext mlContext, IDataView trainingData, IDataView testData, IEstimator<ITransformer> dataPipeline)
    {
        // Initialize LightGBM trainer
        var trainer = mlContext.BinaryClassification.Trainers.LightGbm(
            "Survived",           // Label column name
            "Features",           // Feature column name
            numberOfLeaves: 31,
            minimumExampleCountPerLeaf: 20,
            learningRate: 0.1,
            numberOfIterations: 100);

        var pipeline = dataPipeline.Append(trainer);

        Console.WriteLine("Starting LightGBM training...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var model = pipeline.Fit(trainingData);
        sw.Stop();

        var trainPredictions = model.Transform(trainingData);
        var trainMetrics = mlContext.BinaryClassification.Evaluate(
            trainPredictions,
            labelColumnName: "Survived",
            scoreColumnName: "Score",
            probabilityColumnName: "Probability",
            predictedLabelColumnName: "PredictedLabel");

        var testPredictions = model.Transform(testData);
        SavePredictions(mlContext, testPredictions, testData, "../../../standard_lightgbm_predictions.csv");

        mlContext.Model.Save(model, trainingData.Schema, "../../../standard_lightgbm_model.zip");

        return new ModelResults
        {
            Accuracy = trainMetrics.Accuracy,
            AreaUnderRocCurve = trainMetrics.AreaUnderRocCurve,
            F1Score = trainMetrics.F1Score,
            Type = "Standard LightGBM",
            TrainingTime = sw.ElapsedMilliseconds
        };
    }

    static ModelResults TrainFastTreeLightGBM(MLContext mlContext, IDataView trainingData, IDataView testData, IEstimator<ITransformer> dataPipeline)
    {
        // Initialize FastTree trainer
        var trainer = mlContext.BinaryClassification.Trainers.FastTree(
            "Survived",           // Label column name
            "Features",           // Feature column name
            numberOfLeaves: 31,
            numberOfTrees: 100,
            minimumExampleCountPerLeaf: 20,
            learningRate: 0.1);

        var pipeline = dataPipeline.Append(trainer);

        Console.WriteLine("Starting FastTree training...");
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var model = pipeline.Fit(trainingData);
        sw.Stop();

        var trainPredictions = model.Transform(trainingData);
        var trainMetrics = mlContext.BinaryClassification.Evaluate(
            trainPredictions,
            labelColumnName: "Survived",
            scoreColumnName: "Score",
            probabilityColumnName: "Probability",
            predictedLabelColumnName: "PredictedLabel");

        var testPredictions = model.Transform(testData);
        SavePredictions(mlContext, testPredictions, testData, "../../../fasttree_lightgbm_predictions.csv");

        mlContext.Model.Save(model, trainingData.Schema, "../../../fasttree_lightgbm_model.zip");

        return new ModelResults
        {
            Accuracy = trainMetrics.Accuracy,
            AreaUnderRocCurve = trainMetrics.AreaUnderRocCurve,
            F1Score = trainMetrics.F1Score,
            Type = "FastTree",
            TrainingTime = sw.ElapsedMilliseconds
        };
    }

    static void SavePredictions(MLContext mlContext, IDataView predictions, IDataView testData, string filename)
    {
        var predictionData = mlContext.Data.CreateEnumerable<TitanicPrediction>(predictions, reuseRowObject: false);
        var testDataEnum = mlContext.Data.CreateEnumerable<TitanicTestData>(testData, reuseRowObject: false);

        using (var writer = new StreamWriter(filename))
        {
            writer.WriteLine("PassengerId,Survived");
            foreach (var (pred, original) in predictionData.Zip(testDataEnum))
            {
                writer.WriteLine($"{original.PassengerId},{(pred.Prediction ? 1 : 0)}");
            }
        }
    }

    static void PrintResults(Dictionary<string, object> results)
    {
        Console.WriteLine("\nModel Comparison Results:");
        Console.WriteLine(new string('-', 100));
        Console.WriteLine($"{"Model Type",-20} {"Accuracy",-15} {"AUC",-15} {"F1 Score",-15} {"Training Time",-15}");
        Console.WriteLine(new string('-', 100));

        foreach (var result in results)
        {
            var metrics = (ModelResults)result.Value;
            Console.WriteLine(
                $"{metrics.Type,-20} {metrics.Accuracy:P2,-15} {metrics.AreaUnderRocCurve:P2,-15} " +
                $"{metrics.F1Score:P2,-15} {metrics.TrainingTime / 1000.0:F2}s");
        }
        Console.WriteLine(new string('-', 100));
    }

    class ModelResults
    {
        public required string Type { get; set; }
        public double Accuracy { get; set; }
        public double AreaUnderRocCurve { get; set; }
        public double F1Score { get; set; }
        public long TrainingTime { get; set; }
    }
}