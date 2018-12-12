package ClaMP;

import FuzzyKNN.CrossValidation;
import LocalMeanKNN.LocalMeanKNNClassifier;
import LocalMeanKNN.LocalMeanKNNUtil;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.shu.edu.cn on 2018/11/26.
 * This Class implement "Local Mean KNN" algorithm on ClaMP dataset.
 */
public class ClaMPLocalMeanKNN {
    public static void main(String[] args) {
        try {
            // load data. Please replace the relative address with an Absolute Address here.
            String dataPath = "../..../DataSet/ClaMP_Raw-5184.csv";
            LocalMeanKNNUtil localMeanKNNUtil = new LocalMeanKNNUtil();
            localMeanKNNUtil.loadData(dataPath);
            ArrayList<Float[]> allFeatures = localMeanKNNUtil.getFeatures();
            ArrayList<Integer> allLabels = localMeanKNNUtil.getLabels();

            // Split data into benign and malicious.
            ArrayList<Float[]> benignFeatures = new ArrayList<>();
            ArrayList<Integer> benignLabels = new ArrayList<>();
            ArrayList<Float[]> maliciousFeatures = new ArrayList<>();
            ArrayList<Integer> maliciousLabels = new ArrayList<>();
            for (int i = 0; i < allFeatures.size(); i++) {
                if (allLabels.get(i).equals(1)) {
                    benignFeatures.add(allFeatures.get(i));
                    benignLabels.add(1);
                } else {
                    maliciousFeatures.add(allFeatures.get(i));
                    maliciousLabels.add(0);
                }
            }

            String resultFile = "../..../Result/ClaMP/Local_Mean_KNN_ClaMP_Result(k=68).csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            // Notice: "Mean Local KNN" algorithm can only get the prediction label, the probability of each category cannot be obtained.
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel"});
            writer.flush();

            // Cross Validation.
            int fold = 10;
            CrossValidation benignCrossValidation = new CrossValidation(benignFeatures.size(), fold);
            CrossValidation maliciousCrossValidation = new CrossValidation(maliciousLabels.size(), fold);
            for (int i = 0; i < fold; i++) {
                ArrayList<Float[]> trainFeatures = new ArrayList<>();
                ArrayList<Integer> trainLabels = new ArrayList<>();
                ArrayList<Float[]> testFeatures = new ArrayList<>();
                ArrayList<Integer> testLabels = new ArrayList<>();

                for (int j = 0; j < benignCrossValidation.train[i].length; j++) {
                    trainFeatures.add(benignFeatures.get(benignCrossValidation.train[i][j]));
                    trainLabels.add(benignLabels.get(benignCrossValidation.train[i][j]));
                }
                for (int j = 0; j < benignCrossValidation.test[i].length; j++) {
                    testFeatures.add(benignFeatures.get(benignCrossValidation.test[i][j]));
                    testLabels.add(benignLabels.get(benignCrossValidation.test[i][j]));
                }
                for (int j = 0; j < maliciousCrossValidation.train[i].length; j++) {
                    trainFeatures.add(maliciousFeatures.get(maliciousCrossValidation.train[i][j]));
                    trainLabels.add(maliciousLabels.get(maliciousCrossValidation.train[i][j]));
                }
                for (int j = 0; j < maliciousCrossValidation.test[i].length; j++) {
                    testFeatures.add(maliciousFeatures.get(maliciousCrossValidation.test[i][j]));
                    testLabels.add(maliciousLabels.get(maliciousCrossValidation.test[i][j]));
                }

                // fit classifier.
                int k = 68; // k = sqrt(0.9 * 5184) = 68
                LocalMeanKNNClassifier classifier = new LocalMeanKNNClassifier(trainFeatures, trainLabels, k);

                // predict
                ArrayList<Integer> result = classifier.predict(testFeatures);

                String[] record = new String[2];
                for (int j = 0; j < result.size(); j++) {
                    record[0] = String.valueOf(testLabels.get(j));
                    record[1] = String.valueOf(result.get(j));
                    writer.writeRecord(record);
                    writer.flush();
                }
            }
            writer.flush();
            writer.close();
            System.out.println("Finish");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
