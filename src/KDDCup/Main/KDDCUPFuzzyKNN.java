package KDDCup.Main;

import FuzzyKNN.*;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.text.MessageFormat;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 * This Class implement classification and  prediction of Fuzzy KNN algorithm on KDDCUP dataset.
 */
public class KDDCUPFuzzyKNN {
    public static void main(String[] args) {
        FuzzyKNNUtil util = new FuzzyKNNUtil();
        // load training data. Replace the relative address with an Absolute Address here.
        String dataPath = "../..../DataSet/kddcup_data_10_percent_corrected.csv";
        util.loadData(dataPath);
        ArrayList<Float[]> data = FuzzyKNNUtil.getData();
        ArrayList<Float> label = FuzzyKNNUtil.getLabel();
        ArrayList<Range> ranges = util.getRange();
        // display the range of each features.
        for (Range range : ranges) {
            //System.out.println(range.min + " - " + range.max);
        }
        // Initialize config array
        int inputArgNum = data.get(0).length;
        // the number of fuzzy interval on each input feature.
        int[] fuzzyIntervalNumArray = {20, 3, 69, 11, 8, 100, 2, 4, 4, 31, 6, 2, 20, 2, 3, 30, 29, 3, 9, 8, 8, 2, 20, 20, 2, 2, 2, 2, 2, 2, 2, 20, 20, 2, 2, 2, 2, 2, 2, 2, 2};

        // get the vector of training data( including fuzzy interval vector and membership vector).
        ArrayList<Vector> trainVectors = new ArrayList<>();
        ArrayList<Float> trainLabels = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            int[] fuzzySetVector = new int[inputArgNum];
            float[] membershipVector = new float[inputArgNum];
            for (int j = 0; j < inputArgNum; j++) {
                Membership membership = util.getMembership(ranges.get(j), fuzzyIntervalNumArray[j], data.get(i)[j]);
                fuzzySetVector[j] = membership.getFuzzySetNum();
                membershipVector[j] = membership.getMembership();
            }
            Vector temp = new Vector(fuzzySetVector, membershipVector);
            trainVectors.add(temp);
            trainLabels.add(label.get(i));
        }
        System.out.println("train data size = " + trainVectors.size());
        // fit model. k = 3
        FuzzyKNNClassifier classifier = new FuzzyKNNClassifier(trainVectors, trainLabels, 3);

        // load testing data.
        util.setData(new ArrayList<>());
        util.setLabel(new ArrayList<>());
        String testDataPath = "../../DataSet/corrected.csv";
        util.loadData(testDataPath);
        ArrayList<Float[]> testData = FuzzyKNNUtil.getData();
        ArrayList<Float> testLabel = FuzzyKNNUtil.getLabel();
        ArrayList<Vector> testVectors = new ArrayList<>();
        // get the vector of test data( including fuzzy set vector and membership vector).
        for (int i = 0; i < testData.size(); i++) {
            int[] fuzzySetVector = new int[inputArgNum];
            float[] membershipVector = new float[inputArgNum];
            for (int j = 0; j < inputArgNum; j++) {
                Membership membership = util.getMembership(ranges.get(j), fuzzyIntervalNumArray[j], testData.get(i)[j]);
                fuzzySetVector[j] = membership.getFuzzySetNum();
                membershipVector[j] = membership.getMembership();
            }
            testVectors.add(new Vector(fuzzySetVector, membershipVector));
        }
        System.out.println("test data size = " + testVectors.size());
        // predict
        ArrayList<Result> results = classifier.predict(testVectors);

        // Statistic and Analysis.
        int correctPredictCounter = 0;
        for (int i = 0; i < results.size(); i++)
            if (results.get(i).label.equals(testLabel.get(i)))
                correctPredictCounter++;
        System.out.println("Accuracy = " + correctPredictCounter * 1.0 / results.size());

        // save the predict result
        String predictResultPath = MessageFormat.format("../..../Result/KDDCUP/Fuzzy_KNN_KDDCUP_Result(k=3).csv", 3);
        CsvWriter writer = new CsvWriter(predictResultPath, ',', Charset.forName("UTF-8"));
        try {
            /**
             *  PredictLabelProbability : probability of predict label.
             *  RealLabelProbability : the probability of real label.
             *  AllProbability : all the probability on each category.
             */
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel", "PredictLabelProbability", "RealLabelProbability", "AllProbability"});
            String[] temp = new String[5];
            for (int index = 0; index < results.size(); index++) {
                temp[0] = String.valueOf((testLabel.get(index).floatValue()));
                temp[1] = String.valueOf(results.get(index).label.floatValue());
                int predictLabelIndex = (int) results.get(index).label.floatValue();
                String probability = String.valueOf(results.get(index).prohibit[predictLabelIndex]);
                temp[2] = probability;
                int realLabelIndex = (int) testLabel.get(index).floatValue();
                temp[3] = String.valueOf(results.get(index).prohibit[realLabelIndex]);
                String probStr = "";
                for (int l = 0; l < results.get(index).prohibit.length; l++)
                    probStr += results.get(index).prohibit[l].floatValue() + ",";

                probStr = probStr.substring(0, probStr.length() - 1);
                temp[4] = probStr;
                writer.writeRecord(temp);
                writer.flush();
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Finish");
    }
}
