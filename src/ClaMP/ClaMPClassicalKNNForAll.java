package ClaMP;

import FuzzyKNN.CrossValidation;
import FuzzyKNN.FuzzyKNNUtil;
import FuzzyKNN.ModelEvaluator;
import FuzzyKNN.Result;
import com.csvreader.CsvWriter;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/16.
 * This class is used to get the performance (including mean Accuracy and mean LogLoss) of Classical KNN classifier under all legal 'k' (1-4466) on ClaMP data set.
 */
public class ClaMPClassicalKNNForAll {
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
            String resultFile = "../..../Result/Classical_KNN_Predict_Result_Evaluate_For_All_Legal_K.csv";
            CsvWriter writer = new CsvWriter(resultFile, ',', Charset.forName("UTF-8"));
            writer.writeRecord(new String[]{"k", "mean accuracy", "mean log loss"});
            writer.flush();

            int trainDataNum = (int) (0.9 * instances.size());
            for (int k = 1; k < trainDataNum; k++) {
                int fold = 10;
                CrossValidation benignCrossValidation = new CrossValidation(benignInstances.size(), fold);
                CrossValidation maliciousCrossValidation = new CrossValidation(maliciousInstances.size(), fold);

                ArrayList<ModelEvaluator> evaluators = new ArrayList<>();
                FuzzyKNNUtil util = new FuzzyKNNUtil();

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

                    Classifier IBkClassifier = new IBk(k);
                    IBkClassifier.buildClassifier(trainInstances);
                    int sizeOfTestSet = testInstances.numInstances();

                    ArrayList<Float> testLabel = new ArrayList<>();
                    ArrayList<Result> results = new ArrayList<>();
                    for (int j = 0; j < sizeOfTestSet; j++) {
                        testLabel.add((float) testInstances.instance(j).classValue());
                        Float predictLabel = (float) IBkClassifier.classifyInstance(testInstances.instance(j));
                        double[] probabilities = IBkClassifier.distributionForInstance(testInstances.instance(j));
                        Float[] prohibit = new Float[probabilities.length];
                        for (int index = 0; index < probabilities.length; index++)
                            prohibit[index] = (float) probabilities[index];
                        Result result = new Result(predictLabel, prohibit);
                        results.add(result);
                    }
                    evaluators.add(util.evaluateModel(results, testLabel));
                }
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
            writer.flush();
            writer.close();
            System.out.println("Finish");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
