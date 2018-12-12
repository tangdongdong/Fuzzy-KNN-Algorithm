package KDDCup.Main;

import com.csvreader.CsvWriter;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/12.
 * This Class realize classification and  prediction of SVM algorithm on KDDCUP dataset.
 */
public class KDDCUPSVM {
    public static void main(String[] args) {
        try {
            Classifier SVMClassifier = new SMO();
            // load train data. Replace the relative address with an Absolute Address here.
            File trainFile = new File("../..../DataSet/kddcup_data_10_percent_corrected.arff");
            ArffLoader arffLoader = new ArffLoader();
            arffLoader.setFile(trainFile);
            Instances trainInstances = arffLoader.getDataSet();

            // load test data.
            File testFile = new File("../..../DataSet/corrected.arff");
            arffLoader.setFile(testFile);
            Instances testInstances = arffLoader.getDataSet();
            // row index about category.
            trainInstances.setClassIndex(41);
            testInstances.setClassIndex(41);

            SVMClassifier.buildClassifier(trainInstances);
            int sizeOfTestSet = testInstances.numInstances();

            // predict and save result.
            String resultFile = "../..../Result/KDDCUP/SVM_KDDCUP_Result.csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel", "PredictLabelProbability", "RealLabelProbability", "AllProbability"});
            String[] record = new String[5];

            for (int i = 0; i < sizeOfTestSet; i++) {
                int realLabel = (int) testInstances.instance(i).classValue();
                record[0] = String.valueOf(realLabel);
                int predictLabel = (int) SVMClassifier.classifyInstance(testInstances.instance(i));
                record[1] = String.valueOf(predictLabel);

                double[] probabilities = SVMClassifier.distributionForInstance(testInstances.instance(i));
                float predictLabelProbability = (float) probabilities[predictLabel];
                record[2] = String.valueOf(predictLabelProbability);
                float realLabelProbability = (float) probabilities[realLabel];
                record[3] = String.valueOf(realLabelProbability);
                String allProbOnEachClass = "";
                for (double prob : probabilities)
                    allProbOnEachClass += (float) prob + ",";
                allProbOnEachClass = allProbOnEachClass.substring(0, allProbOnEachClass.length() - 1);
                record[4] = allProbOnEachClass;
                writer.writeRecord(record);
                writer.flush();
            }
            writer.close();
            System.out.println("Finish");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
