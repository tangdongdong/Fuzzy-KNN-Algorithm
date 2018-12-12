package FuzzyKNN;

import com.csvreader.CsvReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 */
public class FuzzyKNNUtil {
    private static ArrayList<Float> label = new ArrayList<>();
    private static ArrayList<Float[]> data = new ArrayList<>();
    private static ArrayList<String> featureNames = new ArrayList<>();

    public void loadData(String dataPath) {
        try {
            CsvReader reader = new CsvReader(dataPath);
            // store feature name
            if (reader.readHeaders()) {
                String[] names = reader.getHeaders();
                for (String name : names)
                    featureNames.add(name);
                featureNames.remove(featureNames.size() - 1);
            }
            // store feature value and label
            int i = 0;
            while (reader.readRecord()) {
                String[] valuesAndLabel = reader.getValues();
                try {
                    Float[] valuesAndLabelFloatFormat = stringArr2FloatArr(valuesAndLabel);
                    data.add(Arrays.copyOfRange(valuesAndLabelFloatFormat, 0, valuesAndLabelFloatFormat.length - 1));
                    label.add(valuesAndLabelFloatFormat[valuesAndLabelFloatFormat.length - 1]);
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("i = " + i);
                }
                i++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Range> getRange() {
        if (label.size() == 0 || data.size() == 0 || featureNames.size() == 0) {
            System.err.println("Data is not initialized, please call 'loadData()' function to initialize first.");
            return null;
        }
        ArrayList<Range> ranges = new ArrayList<Range>();
        int featuresNum = data.get(0).length;
        int samplesNum = data.size();
        for (int i = 0; i < featuresNum; i++) {
            float max = Float.MIN_VALUE;
            float min = Float.MAX_VALUE;
            for (int j = 0; j < samplesNum; j++) {
                if (data.get(j)[i] < min)
                    min = data.get(j)[i];
                if (data.get(j)[i] > max)
                    max = data.get(j)[i];
            }
            ranges.add(new Range(min, max));
        }
        return ranges;
    }

    // This function is very important, Cautiously Modify.
    public Membership getMembership(Range range, int setNum, Float value) {
        if (value <= range.min)
            return new Membership(1, 1);
        else if (value >= range.max)
            return new Membership(setNum, 1);

        float step = (range.max - range.min) / (setNum - 1);
        int belongSet = 1;
        float maxMembership = 0f;
        for (int i = 1; i <= setNum; i++) {
            float tempMembership = 0f;
            if (i == 1) {
                if (value <= range.min + step)
                    tempMembership = 1 - ((value - range.min) / step);
            } else if (i == setNum) {
                if (value >= range.max - step)
                    tempMembership = (value - (range.max - step)) / step;
            } else {
                if (value > range.min + (i - 2) * step && value <= range.min + i * step)
                    tempMembership = 1 - Math.abs((range.min + (i - 1) * step) - value) / step;
            }
            if (tempMembership > maxMembership) {
                maxMembership = tempMembership;
                belongSet = i;
            }
        }
        return new Membership(belongSet, maxMembership);
    }

    private int getLabelIndex(Float[] allLabel, Float label) {
        for (int i = 0; i < allLabel.length; i++)
            if (allLabel[i].equals(label))
                return i;
        return -1;
    }

    public static ArrayList<Float> getLabel() {
        return label;
    }

    public static ArrayList<Float[]> getData() {
        return data;
    }

    public synchronized ModelEvaluator evaluateModel(ArrayList<Result> results, ArrayList<Float> testLabel) {
        Float[] allLabel;
        ArrayList<Float> temp = new ArrayList<>();
        for (Float item : testLabel)
            if (!temp.contains(item))
                temp.add(item);
        Collections.sort(temp);
        allLabel = new Float[temp.size()];
        for (int i = 0; i < temp.size(); i++)
            allLabel[i] = temp.get(i);

        int correctPredictedNum = 0;
        float sum = 0.0f;
        for (int i = 0; i < results.size(); i++) {
            Float predictLabel = results.get(i).label;
            Float realLabel = testLabel.get(i);
            if (predictLabel.equals(realLabel)) {
                correctPredictedNum++;
                int index = getLabelIndex(allLabel, realLabel);
                sum += (float) Math.log(results.get(i).prohibit[index]);
            }
        }
        float logLoss = -1.0f * (sum / results.size());
        float accuracy = correctPredictedNum * 1.0f / results.size();
        return new ModelEvaluator(accuracy, logLoss);
    }

    public static ArrayList<String> getFeatureNames() {
        return featureNames;
    }

    public static void setLabel(ArrayList<Float> label) {
        FuzzyKNNUtil.label = label;
    }

    public static void setData(ArrayList<Float[]> data) {
        FuzzyKNNUtil.data = data;
    }

    public static Float[] stringArr2FloatArr(String[] strings) {
        Float[] floats = new Float[strings.length];
        for (int i = 0; i < strings.length; i++) {
            try {
                floats[i] = Float.valueOf(strings[i]);
            } catch (NumberFormatException e) {
                throw e;
            }

        }
        return floats;
    }
}
