package ClaMP;

import FuzzyKNN.CrossValidation;
import com.csvreader.CsvWriter;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/13.
 * This Class implement classification and  prediction of SVM algorithm on ClaMP dataset.
 */
public class ClaMPSVM {
    public static void main(String[] args) {
        try {
            //load data. Replace the relative address with an Absolute Address here.
            File dataFile = new File("../..../DataSet/ClaMP_Raw-5184.arff");
            ArffLoader arffLoader = new ArffLoader();
            arffLoader.setFile(dataFile);
            Instances instances = arffLoader.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);

            // Prepare an empty collection.
            Instances emptyInstances = new Instances(instances);
            emptyInstances.delete();
            // Split data into benign and malicious.
            Instances benignInstances = new Instances(instances);
            // Remove all malicious samples, leaving only benign samples.
            for (int i = benignInstances.size() - 1; i >= 0; i--)
                if ((int) benignInstances.instance(i).classValue() == 1)
                    benignInstances.remove(i);
            // Remove all benign samples, leaving only malicious samples.
            Instances maliciousInstances = new Instances(instances);
            for (int i = maliciousInstances.size() - 1; i >= 0; i--)
                if ((int) maliciousInstances.instance(i).classValue() == 0)
                    maliciousInstances.remove(i);

            String resultFile = "../..../Result/ClaMP/SVM_ClaMP_Result.csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            writer.writeRecord(new String[]{"RealLabel", "PredictLabel", "PredictLabelProbability", "RealLabelProbability", "AllProbability"});
            writer.flush();
            String[] record = new String[5];

            // Cross Validation. Train data size : Test data size = 90% : 10% = 4665 : 519
            /**
             *                Train   Test
             * benign          90%     10%
             * malicious       90%     10%
             * Total sample    4665    519
             */
            int fold = 10;
            CrossValidation benignCrossValidation = new CrossValidation(benignInstances.size(), fold);
            CrossValidation maliciousCrossValidation = new CrossValidation(maliciousInstances.size(), fold);
            for (int i = 0; i < fold; i++) {
                Instances trainInstances = new Instances(emptyInstances);
                Instances testInstances = new Instances(emptyInstances);

                for (int j = 0; j < benignCrossValidation.train[i].length; j++)
                    trainInstances.add(benignInstances.get(benignCrossValidation.train[i][j]));
                for (int j = 0; j < benignCrossValidation.test[i].length; j++)
                    testInstances.add(benignInstances.get(benignCrossValidation.test[i][j]));
                for (int j = 0; j < maliciousCrossValidation.train[i].length; j++)
                    trainInstances.add(maliciousInstances.get(maliciousCrossValidation.train[i][j]));
                for (int j = 0; j < maliciousCrossValidation.test[i].length; j++)
                    testInstances.add(maliciousInstances.get(maliciousCrossValidation.test[i][j]));

                Classifier SVMClassifier = new SMO();
                SVMClassifier.buildClassifier(trainInstances);
                int sizeOfTestSet = testInstances.numInstances();
                for (int j = 0; j < sizeOfTestSet; j++) {
                    int realLabel = (int) testInstances.instance(j).classValue();
                    record[0] = String.valueOf(realLabel);
                    int predictLabel = (int) SVMClassifier.classifyInstance(testInstances.instance(j));
                    record[1] = String.valueOf(predictLabel);
                    double[] probabilities = SVMClassifier.distributionForInstance(testInstances.instance(j));
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
