package FuzzyKNN;

/**
 * Created by mingdongtang@t.shu.edu.cn on 2018/11/10.
 * Represents the fuzzy region vector and membership vector of a sample.
 * Corresponds to Equation (3) in the paper.
 */
public class Vector {
    // fuzzy region vector.
    public int[] fuzzySetVector;
    // membership vector
    public float[] membershipVector;

    public Vector(int[] fuzzySetVector, float[] membershipVector) {
        this.fuzzySetVector = fuzzySetVector;
        this.membershipVector = membershipVector;
    }
}
