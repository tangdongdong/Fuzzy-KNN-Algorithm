package FuzzyKNN;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 */
public class Result {
    /**
     * Indicates the predicted label.
     */
    public Float label;
    /**
     * Indicates the probability that it may belong to each category.
     **/
    public Float[] prohibit;

    public Result(Float label, Float[] prohibit) {
        this.label = label;
        this.prohibit = prohibit;
    }
}
