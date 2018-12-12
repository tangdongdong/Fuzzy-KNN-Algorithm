package LocalMeanKNN;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/26.
 */
public class Neighbor implements Comparable<Neighbor> {
    public int label;
    public float distance;

    public Neighbor(int label, float distance) {
        this.label = label;
        this.distance = distance;
    }

    @Override
    public int compareTo(Neighbor other) {
        if (this.distance < other.distance)
            return -1;
        else if (this.distance > other.distance)
            return 1;
        else
            return 0;
    }
}
