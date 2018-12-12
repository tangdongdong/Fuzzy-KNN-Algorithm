package LocalMeanKNN;

import com.csvreader.CsvReader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/26.
 */
public class LocalMeanKNNUtil {
    private ArrayList<Float[]> features = new ArrayList<>();
    private ArrayList<Integer> labels = new ArrayList<>();

    public void loadData(String dataPath) {
        try {
            CsvReader reader = new CsvReader(dataPath);
            reader.readHeaders(); // read header
            while (reader.readRecord()) {
                String[] valuesAndLabel = reader.getValues();
                Float[] floats = this.Str2Float(valuesAndLabel);
                Float[] feature = new Float[floats.length - 1];
                for (int i = 0; i < feature.length; i++)
                    feature[i] = floats[i];
                features.add(feature);
                labels.add((int) floats[floats.length - 1].floatValue());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<Float[]> getFeatures() {
        return features;
    }

    public ArrayList<Integer> getLabels() {
        return labels;
    }

    private Float[] Str2Float(String[] strArr) {
        Float[] floats = new Float[strArr.length];
        for (int i = 0; i < strArr.length; i++) {
            floats[i] = Float.valueOf(strArr[i]);
        }
        return floats;
    }

}
