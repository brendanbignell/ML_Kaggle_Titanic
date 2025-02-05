using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

public class Program
{
    // Define the data structure for training data (includes Survived column)
    public class TitanicTrainData
    {
        [LoadColumn(0)]
        public float PassengerId;

        [LoadColumn(1)]
        public bool Survived;

        [LoadColumn(2)]
        public float Pclass;

        [LoadColumn(3)]
        public string Name;

        [LoadColumn(4)]
        public string Sex;

        [LoadColumn(5)]
        public float Age;

        [LoadColumn(6)]
        public float SibSp;

        [LoadColumn(7)]
        public float Parch;

        [LoadColumn(8)]
        public string Ticket;

        [LoadColumn(9)]
        public float Fare;

        [LoadColumn(10)]
        public string Cabin;

        [LoadColumn(11)]
        public string Embarked;
    }

    // Define the data structure for test data (no Survived column)
    public class TitanicTestData
    {
        [LoadColumn(0)]
        public float PassengerId;

        [LoadColumn(1)]
        public float Pclass;

        [LoadColumn(2)]
        public string Name;

        [LoadColumn(3)]
        public string Sex;

        [LoadColumn(4)]
        public float Age;

        [LoadColumn(5)]
        public float SibSp;

        [LoadColumn(6)]
        public float Parch;

        [LoadColumn(7)]
        public string Ticket;

        [LoadColumn(8)]
        public float Fare;

        [LoadColumn(9)]
        public string Cabin;

        [LoadColumn(10)]
        public string Embarked;
    }

    // Define the prediction output class
    public class TitanicPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }
    }

    static void Main(string[] args)
    {
        // Create ML.NET context
        var mlContext = new MLContext(seed: 0);

        // Load training data
        Console.WriteLine("Loading training data...");
        IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TitanicTrainData>(
            path: "../../../train.csv",
            hasHeader: true,
            separatorChar: ',');

        // Load test data
        Console.WriteLine("Loading test data...");
        IDataView testDataView = mlContext.Data.LoadFromTextFile<TitanicTestData>(
            path: "../../../test.csv",
            hasHeader: true,
            separatorChar: ',');

        // Create data processing pipeline
        Console.WriteLine("Creating and training model...");
        var pipeline = mlContext.Transforms
            // Convert categorical data to numeric
            .Categorical.OneHotEncoding(outputColumnName: "SexEncoded", inputColumnName: "Sex")
            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "EmbarkedEncoded", inputColumnName: "Embarked"))
            // Replace missing values
            .Append(mlContext.Transforms.ReplaceMissingValues(
                new[] {
                    new InputOutputColumnPair("Age", "Age"),
                    new InputOutputColumnPair("Fare", "Fare")
                }))
            // Combine features into a single column
            .Append(mlContext.Transforms.Concatenate("Features",
                "Pclass", "SexEncoded", "Age", "SibSp", "Parch", "Fare", "EmbarkedEncoded"))
            // Use LightGBM for training
            .Append(mlContext.BinaryClassification.Trainers.LightGbm(
                labelColumnName: "Survived",
                featureColumnName: "Features",
                numberOfLeaves: 31,
                numberOfIterations: 100,
                minimumExampleCountPerLeaf: 20,
                learningRate: 0.1));

        // Train the model
        var model = pipeline.Fit(trainingDataView);

        // Make predictions on training data
        Console.WriteLine("\nEvaluating on training set...");
        var trainPredictions = model.Transform(trainingDataView);
        var trainMetrics = mlContext.BinaryClassification.Evaluate(
            trainPredictions,
            labelColumnName: "Survived",
            scoreColumnName: "Score",
            probabilityColumnName: "Probability",
            predictedLabelColumnName: "PredictedLabel");

        // Print training metrics
        Console.WriteLine("Training Metrics:");
        Console.WriteLine($"Accuracy: {trainMetrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {trainMetrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {trainMetrics.F1Score:P2}");

        // Make predictions on test data
        Console.WriteLine("\nGenerating predictions for test set...");
        var testPredictions = model.Transform(testDataView);

        // Save predictions to a file
        Console.WriteLine("\nSaving predictions...");
        var predictionData = mlContext.Data.CreateEnumerable<TitanicPrediction>(testPredictions, reuseRowObject: false);
        var testData = mlContext.Data.CreateEnumerable<TitanicTestData>(testDataView, reuseRowObject: false);

        using (var writer = new StreamWriter("predictions.csv"))
        {
            writer.WriteLine("PassengerId,Survived");
            foreach (var (pred, original) in predictionData.Zip(testData))
            {
                writer.WriteLine($"{original.PassengerId},{(pred.Prediction ? 1 : 0)}");
            }
        }

        // Save the model
        Console.WriteLine("Saving model...");
        mlContext.Model.Save(model, trainingDataView.Schema, "../../../titanic-model.zip");

        Console.WriteLine("\nModel training and evaluation complete!");
        Console.WriteLine("Predictions saved to 'predictions.csv'");

        // Print feature importance if available
        try
        {
            var featureImportance = trainPredictions.GetColumn<float[]>("FeatureImportance").First();
            var featureNames = new[] { "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked" };

            Console.WriteLine("\nFeature Importance:");
            for (int i = 0; i < featureNames.Length && i < featureImportance.Length; i++)
            {
                Console.WriteLine($"{featureNames[i]}: {featureImportance[i]:F4}");
            }
        }
        catch
        {
            Console.WriteLine("\nFeature importance information not available");
        }
    }
}