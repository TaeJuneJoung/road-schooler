#### Tensor Ranks, Shapes, and Types

-   Rank

몇 차원의 array냐?

| Rank | Math entity | Python example                                               |
| ---- | ----------- | ------------------------------------------------------------ |
| 0    | Scalar      | s = 483                                                      |
| 1    | Vector      | v = [1.1, 2.2, 3.3]                                          |
| 2    | Matrix      | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]                        |
| 3    | 3-Tensor    | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n    | n-Tensor    | ...                                                          |



-   shape

각각의 element에 몇개씩 들어있느냐?

| Rank | Shape              | Dimension number | Example                                 |
| ---- | ------------------ | ---------------- | --------------------------------------- |
| 0    | []                 | 0-D              | A 0-tensor.A scalar.                    |
| 1    | [D0]               | 1-D              | A 1-D tensor with shape [5].            |
| 2    | [D0, D1]           | 2D               | A 2-D tensor with shape [3, 4].         |
| 3    | [D0, D1, D2]       | 3-D              | A 3-D tensor with shape [1, 4, 3].      |
| n    | [D0, D1, ... Dn-1] | n-D              | A tensor with shape [D0, D1, ... Dn-1]. |



-   ```
    t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ```

    shape [3, 3]



-   Type

| Data type | Python type | Description             |
| --------- | ----------- | ----------------------- |
| DT_FLOAT  | tf.float32  | 32 bits floating point. |
| DT_DOUBLE | tf.float64  | 64 bits floating point. |
| DT_INT8   | tf.int8     | 8 bits signed integer.  |
| DT_INT16  | tf.int16    | 16 bits signed integer. |
| DT_INT32  | tf.int32    | 32 bits signed integer. |
| DT_INT64  | tf.int64    | 64 bits signed integer. |

대부분 32를 사용함