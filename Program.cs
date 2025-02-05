using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using System.Text.Json;
using System.Diagnostics;

public class Program
{
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
            .Append(mlContext.Transforms.NormalizeMinMax("Age"))
            .Append(mlContext.Transforms.NormalizeMinMax("Fare"))
            .Append(mlContext.Transforms.NormalizeMinMax("Pclass"))
            .Append(mlContext.Transforms.Concatenate("Features",
                "Pclass", "SexEncoded", "Age", "SibSp", "Parch", "Fare", "EmbarkedEncoded"));

        var results = new Dictionary<string, object>();

        // Train all models
        Console.WriteLine("\nTraining Standard LightGBM...");
        var standardLgbm = TrainStandardLightGBM(mlContext, trainingDataView, testDataView, dataPipeline);
        results["Standard LightGBM"] = standardLgbm;

        Console.WriteLine("\nTraining FastTree...");
        var fastTreeLgbm = TrainFastTreeLightGBM(mlContext, trainingDataView, testDataView, dataPipeline);
        results["FastTree"] = fastTreeLgbm;

        Console.WriteLine("\nTraining Deep Learning Model...");
        var deepLearning = TrainDeepLearning(mlContext, trainingDataView, testDataView, dataPipeline);
        results["Deep Learning"] = deepLearning;

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
        var sw = Stopwatch.StartNew();
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
        var sw = Stopwatch.StartNew();
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

    static ModelResults TrainDeepLearning(MLContext mlContext, IDataView trainingData, IDataView testData, IEstimator<ITransformer> dataPipeline)
    {
        // Enhanced preprocessing pipeline
        var enhancedPipeline = mlContext.Transforms
            .Categorical.OneHotEncoding(outputColumnName: "SexEncoded", inputColumnName: "Sex")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EmbarkedEncoded", inputColumnName: "Embarked"))
            .Append(mlContext.Transforms.ReplaceMissingValues(
                new[] {
                    new InputOutputColumnPair("Age", "Age"),
                    new InputOutputColumnPair("Fare", "Fare")
                }))
            // Enhanced normalization
            .Append(mlContext.Transforms.NormalizeMinMax("Age"))
            .Append(mlContext.Transforms.NormalizeMinMax("Fare"))
            .Append(mlContext.Transforms.NormalizeMinMax("Pclass"))
            .Append(mlContext.Transforms.NormalizeMinMax("SibSp"))
            .Append(mlContext.Transforms.NormalizeMinMax("Parch"))
            // Combine all features
            .Append(mlContext.Transforms.Concatenate("Features",
                "Pclass", "SexEncoded", "Age", "SibSp", "Parch",
                "Fare", "EmbarkedEncoded"))
            // Add training optimizations
            .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
            .AppendCacheCheckpoint(mlContext);

        // Create stacking ensemble
        var trainer1 = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Survived",
                FeatureColumnName = "Features",
                MaximumNumberOfIterations = 1000,
                L2Regularization = 0.00001f,
                L1Regularization = 0.00001f
            });

        var trainer2 = mlContext.BinaryClassification.Trainers.FastForest(
            new FastForestBinaryTrainer.Options
            {
                LabelColumnName = "Survived",
                FeatureColumnName = "Features",
                NumberOfTrees = 200,
                NumberOfLeaves = 50,
                MinimumExampleCountPerLeaf = 10
            });

        var trainer3 = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
            new LbfgsLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = "Survived",
                FeatureColumnName = "Features",
                L1Regularization = 0.00001f,
                L2Regularization = 0.00001f,
                OptimizationTolerance = 1e-8f,
                HistorySize = 100,
                MaximumNumberOfIterations = 1000
            });

        // Create ensemble pipeline with multiple models
        var ensemblePipeline = enhancedPipeline
            .Append(trainer1)
            .Append(trainer2)
            .Append(trainer3)
            .Append(mlContext.BinaryClassification.Calibrators.Platt(labelColumnName: "Survived"));

        // Train model
        Console.WriteLine("Training ensemble model...");
        var sw = Stopwatch.StartNew();

        var model = ensemblePipeline.Fit(trainingData);
        sw.Stop();

        // Evaluate on training data
        var trainPredictions = model.Transform(trainingData);
        var metrics = mlContext.BinaryClassification.Evaluate(
            trainPredictions,
            labelColumnName: "Survived",
            scoreColumnName: "Score",
            probabilityColumnName: "Probability",
            predictedLabelColumnName: "PredictedLabel");

        // Generate predictions for test set
        var testPredictions = model.Transform(testData);
        SavePredictions(mlContext, testPredictions, testData, "../../../deep_learning_predictions.csv");

        // Save the model
        mlContext.Model.Save(model, trainingData.Schema, "../../../deep_learning_model.zip");

        return new ModelResults
        {
            Accuracy = metrics.Accuracy,
            AreaUnderRocCurve = metrics.AreaUnderRocCurve,
            F1Score = metrics.F1Score,
            Type = "Deep Learning",
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
            var accuracy = metrics.Accuracy.ToString("F4").PadRight(15);
            var auc = metrics.AreaUnderRocCurve.ToString("F4").PadRight(15);
            var f1 = metrics.F1Score.ToString("F4").PadRight(15);
            var time = $"{metrics.TrainingTime / 1000.0:F2}s".PadRight(15);

            Console.WriteLine($"{metrics.Type,-20} {accuracy} {auc} {f1} {time}");
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