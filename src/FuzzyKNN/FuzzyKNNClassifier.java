package FuzzyKNN;

import java.util.ArrayList;
import java.util.Collections;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 */
public class FuzzyKNNClassifier {
    private static final long serialVersionUID = -5809782578273944999L;
    // Store the vector(fuzzy region vector and membership vector) of training data.
    private ArrayList<Vector> trainVectors = new ArrayList<>();
    // Store the label of training data.
    private ArrayList<Float> labels = new ArrayList<>();

    // Store the category of training data. e.g. If training data's label = {0,1,2,1,2,3,4}, then allLabel = [0,1,2,3,4].
    // Notice: The label of the training data must be start at 0 and continuous.
    private Float[] allLabel;
    private int k;

    public FuzzyKNNClassifier(ArrayList<Vector> trainVectors, ArrayList<Float> labels, int n_neighbors) {
        if (n_neighbors < 1 || n_neighbors > trainVectors.size()) {
            System.err.println("parameter 'k' is illegal, construct classifier failure.");
            return;
        }
        this.trainVectors = trainVectors;
        this.labels = labels;
        this.k = n_neighbors;
        this.initAllLabel();
    }

    public FuzzyKNNClassifier(ArrayList<Vector> trainVectors, ArrayList<Float> labels) {
        // default n_neighbors = 1
        this(trainVectors, labels, 1);
    }

    public ArrayList<Result> predict(ArrayList<Vector> test) {
        ArrayList<Result> predictResult = new ArrayList<>();
        // predict one by one.
        for (Vector testSample : test) {
            Result result = predictOne(testSample);
            predictResult.add(result);
        }
        return predictResult;
    }

    /**
     * @param testVector test sample's fuzzy vector.
     * @return predict label.
     */
    private Result predictOne(Vector testVector) {
        float predictLabel = -1.0f;

        // Step 1 : Get a list of index of fuzzy region matches. Implement the algorithm 2 in the paper.
        ArrayList<Integer> fuzzySetMatchedIndex = getFuzzySetMatchedIndex(testVector);

        // Step 2 : get the index and distance of the K-nearest neighbors.
        ArrayList<Integer> fuzzySetMatchedIndexClone = deepCloneIntegerList(fuzzySetMatchedIndex);
        Integer[] kNearestNeighborsIndex = new Integer[this.k];
        float[] kNearestNeighborsMembershipDistance = new float[this.k];
        getIndexAndDistanceOfKNearestNeighbors(testVector, fuzzySetMatchedIndexClone, kNearestNeighborsIndex, kNearestNeighborsMembershipDistance);

        // Step 3 : Get the predict Result( predict label and probabilities).
        Float[] sumOfReciprocalOfIndexOnEachCategory = getSumOfReciprocalOfIndexOnEachCategory(kNearestNeighborsIndex);
        Float[] probabilities = getProbabilityOnEachCategory(sumOfReciprocalOfIndexOnEachCategory);
        predictLabel = getPredictLabel(probabilities);
        return new Result(predictLabel, probabilities);
    }

    // Calculate the probability of belonging to each category.
    private Float[] getProbabilityOnEachCategory(Float[] sumOfReciprocalOfIndexOnEachCategory) {
        int categoryNum = this.allLabel.length;
        Float[] probabilities = new Float[categoryNum];
        // init probability.
        for (int i = 0; i < probabilities.length; i++)
            probabilities[i] = 0.0f;

        Float sum = 0.0f;
        for (Float temp : sumOfReciprocalOfIndexOnEachCategory)
            sum += temp;

        for (int i = 0; i < sumOfReciprocalOfIndexOnEachCategory.length; i++)
            probabilities[i] = sumOfReciprocalOfIndexOnEachCategory[i] / sum;
        return probabilities;
    }

    // Get the index and distance of the K-nearest neighbor.
    private void getIndexAndDistanceOfKNearestNeighbors(Vector test, ArrayList<Integer> fuzzyRegionMatchedIndex, Integer[] kNearestNeighborsIndex, float[] kNearestNeighborsMembershipDistance) {
        for (int i = 0; i < this.k; i++) {
            float minDistance = Float.MAX_VALUE;
            int minIndex = 0;
            int removeIndex = 0;
            for (int j = 0; j < fuzzyRegionMatchedIndex.size(); j++) {
                float tempDistance = calcMembershipVectorDistance(trainVectors.get(fuzzyRegionMatchedIndex.get(j)), test);
                if (tempDistance < minDistance) {
                    minDistance = tempDistance;
                    minIndex = fuzzyRegionMatchedIndex.get(j);
                    removeIndex = j;
                }
            }
            kNearestNeighborsIndex[i] = minIndex;
            kNearestNeighborsMembershipDistance[i] = minDistance;
            fuzzyRegionMatchedIndex.remove(removeIndex);
        }
    }

    private float getPredictLabel(Float[] probability) {
        // get the max probability.
        Float maxProbability = probability[0];
        int maxProbabilityIndex = 0;
        for (int i = 0; i < probability.length; i++) {
            if (probability[i] > maxProbability) {
                maxProbability = probability[i];
                maxProbabilityIndex = i;
            }
        }
        return allLabel[maxProbabilityIndex];
    }

    private Float[] getSumOfReciprocalOfIndexOnEachCategory(Integer[] kNearestNeighborsIndex) {
        int categoryNum = allLabel.length;
        Float[] sumOfReciprocalOfIndexOnEachCategory = new Float[categoryNum];
        // init sumOfReciprocalOfIndexOnEachCategory
        for (int i = 0; i < sumOfReciprocalOfIndexOnEachCategory.length; i++)
            sumOfReciprocalOfIndexOnEachCategory[i] = 0.0f;

        for (int i = 0; i < kNearestNeighborsIndex.length; i++) {
            Float realLabel = this.labels.get(kNearestNeighborsIndex[i]);
            for (int j = 0; j < allLabel.length; j++)
                if (realLabel.equals(allLabel[j]))
                    sumOfReciprocalOfIndexOnEachCategory[j] += 1.0f / (i + 1);
        }
        return sumOfReciprocalOfIndexOnEachCategory;
    }

    // Find out the set which with the largest fuzzy region matching, As the algorithm 2 in the paper.
    private ArrayList<Integer> getFuzzySetMatchedIndex(Vector test) {
        ArrayList<Integer> fuzzySetMatchedIndex = new ArrayList<>();
        int unMatch = 0;
        int featureNum = trainVectors.get(0).fuzzySetVector.length;
        while (fuzzySetMatchedIndex.size() < this.k && unMatch <= featureNum) {
            fuzzySetMatchedIndex = new ArrayList<>();
            for (int i = 0; i < trainVectors.size(); i++) {
                if (isFuzzySetMatch(trainVectors.get(i), test, unMatch))
                    fuzzySetMatchedIndex.add(i);
            }
            if (fuzzySetMatchedIndex.size() >= this.k)
                break;
            unMatch++;
        }
        return fuzzySetMatchedIndex;
    }

    /**
     * @param maxNumberMismatches: maximum number of allowed that mismatched fuzzy intervals.
     * @return true:matched;  false:unmatched.
     */
    private boolean isFuzzySetMatch(Vector train, Vector test, int maxNumberMismatches) {
        // If the lengths of the two vectors are not equal, return false.
        if (train.fuzzySetVector.length != test.fuzzySetVector.length) {
            System.err.println("Error: the lengths of the two vectors are not equal.");
            return false;
        }
        int unMatchCounter = 0;
        for (int i = 0; i < train.fuzzySetVector.length; i++)
            if (train.fuzzySetVector[i] != test.fuzzySetVector[i])
                unMatchCounter++;
        if (unMatchCounter <= maxNumberMismatches)
            return true;
        return false;
    }

    // This function is very important, Cautiously Modify.
    private float calcMembershipVectorDistance(Vector train, Vector test) {
        double distance = 0.0f;
        for (int i = 0; i < train.membershipVector.length; i++) {
            distance += Math.pow(((train.fuzzySetVector[i] - test.fuzzySetVector[i]) * 0.5 + Math.abs(train.membershipVector[i] - test.membershipVector[i])), 2);
        }
        return (float) Math.sqrt(distance);
    }

    private ArrayList<Integer> deepCloneIntegerList(ArrayList<Integer> original) {
        ArrayList<Integer> clone = new ArrayList<>();
        if (original == null)
            return clone;
        for (Integer item : original)
            clone.add(item);
        return clone;
    }

    private void initAllLabel() {
        ArrayList<Float> temp = new ArrayList<>();
        for (Float item : this.labels)
            if (!temp.contains(item))
                temp.add(item);
        Collections.sort(temp);
        allLabel = new Float[temp.size()];
        for (int i = 0; i < temp.size(); i++)
            allLabel[i] = temp.get(i);
    }
}
