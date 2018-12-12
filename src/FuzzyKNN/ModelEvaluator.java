package FuzzyKNN;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 * This class is used to store the results of model evaluation, including Accuracy and logLoss.
 */
public class ModelEvaluator {
    private float accuracy;
    // the equation of logLoss, please ref : https://en.wikipedia.org/wiki/Jackknife_variance_estimates_for_random_forest#Examples
    private float logLoss;

    public ModelEvaluator(float accuracy, float logLoss) {
        this.accuracy = accuracy;
        this.logLoss = logLoss;
    }

    public float getAccuracy() {
        return accuracy;
    }

    public float getLogLoss() {
        return logLoss;
    }

    public void display() {
        System.out.println("Accuracy = " + String.format("%.5f", accuracy) + " and logLoss = " + String.format("%.5f", logLoss));
    }
}
