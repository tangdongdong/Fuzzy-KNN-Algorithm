package FuzzyKNN;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 */
public class Membership {
    // the Index of fuzzy region.
    private int fuzzySetNum;
    // the value of membership.
    private float membership;

    public Membership(int fuzzySetNum, float membership) {
        this.fuzzySetNum = fuzzySetNum;
        this.membership = membership;
    }

    public int getFuzzySetNum() {
        return fuzzySetNum;
    }

    public float getMembership() {
        return membership;
    }
}
