package KDDCup.Main;

import LocalMeanKNN.LocalMeanKNNClassifier;
import LocalMeanKNN.LocalMeanKNNUtil;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/27.
 * This Class implement "Local Mean KNN" algorithm on KDDCUP dataset.
 */
public class KDDCUPLocalMeanKNN {
    public static void main(String[] args) {
        try {
            // Step 1: load  data. Replace the relative address with an Absolute Address here.
            String trainDataPath = "../..../DataSet/kddcup_data_10_percent_corrected.csv";
            LocalMeanKNNUtil localMeanKNNUtil = new LocalMeanKNNUtil();
            localMeanKNNUtil.loadData(trainDataPath);
            ArrayList<Float[]> trainFeatures = localMeanKNNUtil.getFeatures();
            ArrayList<Integer> trainLabels = localMeanKNNUtil.getLabels();

            String testDataPath = "../..../DataSet/corrected.csv";
            localMeanKNNUtil = new LocalMeanKNNUtil();
            localMeanKNNUtil.loadData(testDataPath);
            ArrayList<Float[]> testFeatures = localMeanKNNUtil.getFeatures();
            ArrayList<Integer> testLabels = localMeanKNNUtil.getLabels();

            // Step 2: fit model --> predict --> save result.
            String resultFile = "../..../Result/KDDCUP/Local_Mean_KNN_KDDCUP_Result(k=3).csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            // Notice: "Mean Local KNN" algorithm can only get the prediction label, the probabilities on each category cannot be obtained.
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel"});
            writer.flush();

            // fit classifier.
            int k = 3;
            LocalMeanKNNClassifier classifier = new LocalMeanKNNClassifier(trainFeatures, trainLabels, k);

            // predict
            ArrayList<Integer> result = classifier.predict(testFeatures);

            // save result
            String[] record = new String[2];
            for (int j = 0; j < result.size(); j++) {
                record[0] = String.valueOf(testLabels.get(j));
                record[1] = String.valueOf(result.get(j));
                writer.writeRecord(record);
                writer.flush();
            }
            writer.flush();
            writer.close();
            System.out.println("Finish");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
