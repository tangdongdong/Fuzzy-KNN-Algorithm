package ClaMP;

import FuzzyKNN.*;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 * This class is used to get the performance (including mean Accuracy and mean logLoss) of Fuzzy KNN classifier under all legal 'k' (1-4466) on ClaMP data set.
 */
public class ClaMPFuzzyKNNForAllK {
    public static void main(String[] args) throws IOException {
        // load data
        FuzzyKNNUtil util = new FuzzyKNNUtil();
        // Replace the relative address with an Absolute Address here.
        String dataPath = "../..../DataSet/ClaMP_Raw-5184.csv";
        util.loadData(dataPath);
        ArrayList<Float[]> data = FuzzyKNNUtil.getData();
        ArrayList<Float> label = FuzzyKNNUtil.getLabel();
        ArrayList<Range> ranges = util.getRange();

        // Initialize config array
        int intervalNum = 5;
        int inputArgNum = data.get(0).length;
        int[] fuzzySetNumArray = new int[inputArgNum];
        for (int i = 0; i < inputArgNum; i++)
            fuzzySetNumArray[i] = intervalNum;

        // Split data into benign and malicious.
        ArrayList<Vector> benignVectors = new ArrayList<>();
        ArrayList<Float> benignLabels = new ArrayList<>();
        ArrayList<Vector> maliciousVectors = new ArrayList<>();
        ArrayList<Float> maliciousLabels = new ArrayList<>();

        for (int i = 0; i < data.size(); i++) {
            int[] fuzzySetVector = new int[inputArgNum];
            float[] membershipVector = new float[inputArgNum];
            for (int j = 0; j < inputArgNum; j++) {
                Membership membership = util.getMembership(ranges.get(j), fuzzySetNumArray[j], data.get(i)[j]);
                fuzzySetVector[j] = membership.getFuzzySetNum();
                membershipVector[j] = membership.getMembership();
            }
            Vector temp = new Vector(fuzzySetVector, membershipVector);
            if (label.get(i) == 0.0) {
                benignVectors.add(temp);
                benignLabels.add(0.0f);
            } else {
                maliciousVectors.add(temp);
                maliciousLabels.add(1.0f);
            }
        }

        // Train data size : Test data size = 90% : 10% = 4665 : 519
        int trainDataNum = (int) (0.9 * data.size());
        String predictResultEvaluatePath = "../..../Result/ClaMP/Fuzzy_KNN_Predict_Result_Evaluate_For_All_Legal_K.csv";
        CsvWriter writer = new CsvWriter(predictResultEvaluatePath, ',', Charset.forName("UTF-8"));
        // write header.
        writer.writeRecord(new String[]{"k", "mean accuracy", "mean log loss"});
        for (int k = 1; k < trainDataNum; k++) {
            int fold = 10;
            CrossValidation benignCrossValidation = new CrossValidation(benignVectors.size(), fold);
            CrossValidation maliciousCrossValidation = new CrossValidation(maliciousVectors.size(), fold);
            ArrayList<ModelEvaluator> evaluators = new ArrayList<>();
            //                Train   Test
            // benign          90%     10%
            // malicious       90%     10%
            // Total sample    4665    519
            for (int i = 0; i < fold; i++) {
                ArrayList<Vector> trainVector = new ArrayList<>();
                ArrayList<Float> trainLabel = new ArrayList<>();
                ArrayList<Vector> testVector = new ArrayList<>();
                ArrayList<Float> testLabel = new ArrayList<>();
                for (int j = 0; j < benignCrossValidation.train[i].length; j++) {
                    trainVector.add(benignVectors.get(benignCrossValidation.train[i][j]));
                    trainLabel.add(benignLabels.get(benignCrossValidation.train[i][j]));
                }
                for (int j = 0; j < benignCrossValidation.test[i].length; j++) {
                    testVector.add(benignVectors.get(benignCrossValidation.test[i][j]));
                    testLabel.add(benignLabels.get(benignCrossValidation.test[i][j]));
                }
                for (int j = 0; j < maliciousCrossValidation.train[i].length; j++) {
                    trainVector.add(maliciousVectors.get(maliciousCrossValidation.train[i][j]));
                    trainLabel.add(maliciousLabels.get(maliciousCrossValidation.train[i][j]));
                }
                for (int j = 0; j < maliciousCrossValidation.test[i].length; j++) {
                    testVector.add(maliciousVectors.get(maliciousCrossValidation.train[i][j]));
                    testLabel.add(maliciousLabels.get(maliciousCrossValidation.train[i][j]));
                }

                // fit knn model
                FuzzyKNNClassifier classifier = new FuzzyKNNClassifier(trainVector, trainLabel, k);
                // predict
                ArrayList<Result> predict = classifier.predict(testVector);
                // evaluate model
                evaluators.add(util.evaluateModel(predict, testLabel));
            }
            // calculate the mean value about the 10 knn classifier.
            float accuracySum = 0.0f;
            float logLogSum = 0.0f;
            for (ModelEvaluator evaluator : evaluators) {
                accuracySum += evaluator.getAccuracy();
                logLogSum += evaluator.getLogLoss();
            }
            float averageAccuracy = accuracySum / fold;
            float averageLogLoss = logLogSum / fold;
            String[] record = {String.valueOf(k), String.valueOf(averageAccuracy), String.valueOf(averageLogLoss)};
            writer.writeRecord(record);
            writer.flush();
            String content = "k = " + k + ", mean accuracy = " + accuracySum / fold + ", mean logLoss = " + logLogSum / fold + "\n";
            System.out.print(content);
        }
        writer.close();
        System.out.println("Finish");
    }
}
