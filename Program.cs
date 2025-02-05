using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System.Data;

public class Program
{
    // Define the data structure that matches your CSV file
    public class TitanicData
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

        // Load data
        IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TitanicData>(
            path: "../../../train.csv",
            hasHeader: true,
            separatorChar: ',');

        // Create data processing pipeline
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
        Console.WriteLine("Training the model...");
        var model = pipeline.Fit(trainingDataView);

        // Make predictions on training data
        var predictions = model.Transform(trainingDataView);

        // Evaluate the model
        var metrics = mlContext.BinaryClassification.Evaluate(predictions,
            labelColumnName: "Survived",
            scoreColumnName: "Score",
            probabilityColumnName: "Probability",
            predictedLabelColumnName: "PredictedLabel");

        // Print metrics
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

        // Create prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<TitanicData, TitanicPrediction>(model);

        // Example prediction
        var samplePassenger = new TitanicData
        {
            Pclass = 3,
            Sex = "female",
            Age = 26,
            SibSp = 0,
            Parch = 0,
            Fare = 7.925f,
            Embarked = "S"
        };

        var prediction = predictionEngine.Predict(samplePassenger);
        Console.WriteLine($"\nSample Prediction:");
        Console.WriteLine($"Predicted survival: {prediction.Prediction}");
        Console.WriteLine($"Probability: {prediction.Probability:P2}");

        // Save the model
        mlContext.Model.Save(model, trainingDataView.Schema, "../../../titanic-model.zip");
    }
}