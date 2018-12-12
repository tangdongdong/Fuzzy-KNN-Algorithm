package ClaMP;

import FuzzyKNN.*;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/13.
 * This Class implement classification and  prediction of Fuzzy KNN (k=67) algorithm on ClaMP dataset.
 */
public class ClaMPFuzzyKNN {
    public static void main(String[] args) {
        try {
            // load data. Replace the relative address with an Absolute Address here.
            FuzzyKNNUtil util = new FuzzyKNNUtil();
            String dataPath = "../..../DataSet/ClaMP_Raw-5184.csv";
            util.loadData(dataPath);
            ArrayList<Float[]> data = FuzzyKNNUtil.getData();
            ArrayList<Float> label = FuzzyKNNUtil.getLabel();
            ArrayList<Range> ranges = util.getRange();

            // Initialize config array.
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

            String resultFile = "../..../Result/ClaMP/Fuzzy_KNN_ClaMP_Result(k=68).csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel", "PredictLabelProbability", "RealLabelProbability", "AllProbability"});
            writer.flush();

            // Cross Validation.
            int fold = 10;
            CrossValidation benignCrossValidation = new CrossValidation(benignVectors.size(), fold);
            CrossValidation maliciousCrossValidation = new CrossValidation(maliciousVectors.size(), fold);

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
                    testVector.add(maliciousVectors.get(maliciousCrossValidation.test[i][j]));
                    testLabel.add(maliciousLabels.get(maliciousCrossValidation.test[i][j]));
                }

                // fit Fuzzy KNN Classifier.n_neighbor
                int k = 68; //
                FuzzyKNNClassifier classifier = new FuzzyKNNClassifier(trainVector, trainLabel, k);
                // predict
                ArrayList<Result> predict = classifier.predict(testVector);
                // save result.
                String[] record = new String[5];
                for (int index = 0; index < predict.size(); index++) {
                    int realLabel = (int) testLabel.get(index).floatValue();
                    record[0] = String.valueOf(realLabel);
                    int predictLabel = (int) predict.get(index).label.floatValue();
                    record[1] = String.valueOf(predictLabel);
                    Float[] probabilities = predict.get(index).prohibit;
                    float predictLabelProbability = probabilities[predictLabel].floatValue();
                    record[2] = String.valueOf(predictLabelProbability);
                    float realLabelProbability = probabilities[realLabel].floatValue();
                    record[3] = String.valueOf(realLabelProbability);
                    String allPro = "";
                    for (Float pro : probabilities)
                        allPro += pro.floatValue() + ",";
                    allPro = allPro.substring(0, allPro.length() - 1);
                    record[4] = allPro;
                    writer.writeRecord(record);
                    writer.flush();
                }
            }
            writer.close();
            System.out.println("Finsih");
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
