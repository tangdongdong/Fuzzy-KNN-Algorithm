package LocalMeanKNN;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/26.
 * This class is used to implement the "Local Mean KNN" algorithm.
 * Ref : An Improved KNN Algorithm for Imbalanced Data Based on Local Mean.
 * Link : https://www.researchgate.net/publication/288716151_An_improved_KNN_algorithm_for_imbalanced_data_based_on_local_mean
 */

public class LocalMeanKNNClassifier {
    public ArrayList<Float[]> trainFeatures;
    public ArrayList<Integer> trainLabels;
    private ArrayList<Integer> allLabels;
    int k;

    public LocalMeanKNNClassifier(ArrayList<Float[]> trainFeatures, ArrayList<Integer> trainLabels, int k) {
        this.trainFeatures = trainFeatures;
        this.trainLabels = trainLabels;
        this.k = k;
        initAllLabels();
    }

    public LocalMeanKNNClassifier(ArrayList<Float[]> trainFeatures, ArrayList<Integer> trainLabels) {
        this(trainFeatures, trainLabels, 1);
    }

    public ArrayList<Integer> predict(ArrayList<Float[]> testData) {
        ArrayList<Integer> result = new ArrayList<>();
        for (Float[] testSample : testData)
            result.add(predictOne(testSample));

        return result;
    }

    private Integer predictOne(Float[] testSample) {
        // Step 1: find the K-nearest neighbors.
        ArrayList<Neighbor> nearestNeighbors = new ArrayList<>();
        for (int i = 0; i < this.trainFeatures.size(); i++) {
            float distance = calcDistance(this.trainFeatures.get(i), testSample);
            int label = this.trainLabels.get(i);
            // If the size < k, Include this training sample in the set of K-nearest neighbors directly.
            if (nearestNeighbors.size() < this.k) {
                Neighbor neighbor = new Neighbor(label, distance);
                nearestNeighbors.add(neighbor);
            } else {
                Collections.sort(nearestNeighbors);
                // Delete farthest neighbor, and include this training sample in the set of K-nearest neighbors.
                if (distance < nearestNeighbors.get(this.k - 1).distance) {
                    nearestNeighbors.remove(this.k - 1);
                    Neighbor neighbor = new Neighbor(label, distance);
                    nearestNeighbors.add(neighbor);
                }
            }
        }

        // Step 2: calculate the mean distance of each category. and take the class which has the minimum mean distance as the predict label.
        float minMeanDistance = Float.MAX_VALUE;
        int predictLabel = this.allLabels.get(0);
        for (int i = 0; i < this.allLabels.size(); i++) {
            int label = this.allLabels.get(i);

            int sampleCounter = 0;
            float distanceSum = 0.0f;
            for (Neighbor neighbor : nearestNeighbors) {
                if (neighbor.label == label) {
                    sampleCounter++;
                    distanceSum += neighbor.distance;
                }
            }
            if (sampleCounter == 0)
                continue;
            else {
                float meanDistance = distanceSum / sampleCounter;
                if (meanDistance < minMeanDistance) {
                    minMeanDistance = meanDistance;
                    predictLabel = label;
                }
            }
        }

        return predictLabel;
    }

    private float calcDistance(Float[] trainSample, Float[] testSample) {
        float distance = 0.0f;
        for (int i = 0; i < trainSample.length; i++)
            distance += Math.pow(trainSample[i] - testSample[i], 2);
        distance = (float) Math.sqrt(distance);
        return distance;
    }


    /**
     * Get all the labels that appear in the training data, and sort them.
     */
    private void initAllLabels() {
        allLabels = new ArrayList<>();
        for (Integer label : this.trainLabels) {
            if (allLabels.contains(label))
                continue;
            else
                allLabels.add(label);
        }
        Collections.sort(allLabels);
    }
}
