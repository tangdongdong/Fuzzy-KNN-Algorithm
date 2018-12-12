package FuzzyKNN;

/*******************************************************************************
 * Ref : https://github.com/haifengl/smile/blob/master/core/src/main/java/smile/validation/CrossValidation.java
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

public class CrossValidation {
    // Number of folds.
    public final int k;
    public final int[][] train;
    public final int[][] test;

    public CrossValidation(int n, int k) {
        if (n < 0)
            throw new IllegalArgumentException("Invalid sample size: " + n);
        if (k < 0 || k > n)
            throw new IllegalArgumentException("Invalid number of fold: " + k);

        this.k = k;
        int[] index = new int[n];
        for (int i = 0; i < n; i++)
            index[i] = i;

        train = new int[k][];
        test = new int[k][];
        int chunk = n / k;
        for (int i = 0; i < k; i++) {
            int start = chunk * i;
            int end = chunk * (i + 1);
            if (i == k - 1) end = n;

            train[i] = new int[n - end + start];
            test[i] = new int[end - start];
            for (int j = 0, p = 0, q = 0; j < n; j++) {
                if (j >= start && j < end) {
                    test[i][p++] = index[j];
                } else {
                    train[i][q++] = index[j];
                }
            }
        }
    }
}
