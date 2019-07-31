use std::ops::*;
use std::convert::TryInto;
use std::fmt;
use std::cmp;

macro_rules! mat {
    ($($( $x:expr ), *); *) => {
        {
            let mut mat = Matrix::new_empty(Default::default(),Default::default());
            $(
                let mut v = Vec::new();
                $(
                    v.push($x);
                )*
                mat.add_row(&mut v);
            )*
            mat
        }
    };
}

#[allow(dead_code, non_snake_case)]
#[derive(Debug)]
pub struct Matrix<T> {
    M: usize,
    N: usize,
    data: Vec<T>,
}

struct SquaredMatrix<T> {
    M: usize,
    N: usize,
    data: Vec<T>,
}

#[repr(C)]
struct Minor<T> {
    coef: T,
    matrix: Matrix<T>,
}

#[allow(dead_code)]
pub struct Perceptron<T> {
    inputs: Vec<T>,
    weights: Vec<T>,
    bias: T,
    integration_fn: fn(inputs: &Vec<T>, weights: &Vec<T>, bias: &T) -> T,
    activation_fn: fn(integration_result: T) -> T,
}

pub trait Output<T> {
    fn output(&self) -> T;
}

pub trait Dot<T> {
    fn dot(&self, other: &Self) -> T;
}

pub trait VectorVectorAdd<T> {
    fn vvadd(&self, other: &Self) -> Self;
}

pub trait VectorScalarAdd<T> {
    fn vsadd(&self, other: T) -> Self;
}

pub trait ScalarVectorAdd<T> {
    fn vsadd(&self, other: &Vec<T>) -> Vec<T>;
}

pub trait VectorScalarMul<T> {
    fn vsmul(&self, other: T) -> Self;
}

pub trait ScalarVectorMul<T> {
    fn vsmul(&self, other: &Vec<T>) -> Vec<T>;
}

pub trait MatrixMatrixMul<T> {
    fn mul(&self, other: &Matrix<T>) -> Self;
}

pub trait Determinant<T> {
    fn det(m: &Matrix<T>) -> T {
        panic!("trait Determinant not implemented for type T");
    }
}

trait MinorDeterminant<T> {
    fn det(m: &Minor<T>) -> T {
        panic!("trait Determinant not implemented for type T");
    }
}

pub trait PushMultiple<T> {
    fn push_multiple(&mut self, data: &[T]);
}

impl<T> PushMultiple<T> for Vec<T> where
T: Copy {
    fn push_multiple(&mut self, data: &[T]) {
        self.reserve(data.len());
        for idx in 0..data.len() {
            self.push(data[idx]);
        }
    }
}

impl<T: Mul<Output = T> + Sub<Output = T>> Determinant<T> for Matrix<T> where 
T: Default + Add + AddAssign + Sub + SubAssign + Mul + MulAssign + Copy + Neg {
    fn det(m: &Matrix<T>) -> T { 
        /*** It's also a gun ***/
        let mut result = Default::default();

        match m.M {
            1 => {
                result = m[0][0];
            },
            2 => {
                result = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            },
            3 => {
                for i in 0..m.M {
                    let mut a = m[0][i];
                    let mut b = m[0][(m.N-1)-i];
                    for j in 1..m.N {
                        let idx1 = (i + j) % m.M;
                        let idx2 = (((2 * m.N) - 1) - i - j) % m.N;
                        a *= m[j][idx1];
                        b *= m[j][idx2];
                    }   
                    result += a - b;
                }   
            },
            _ => {
                let minors_vec = m.get_minors();
                for i in 0..minors_vec.len() {
                    let det = minors_vec[i].coef * Matrix::det(&minors_vec[i].matrix);
                    if i % 2 == Default::default() {
                        result += det;
                    } else {
                        result -= det;
                    }
                }
            },
        }
        result
    }
}

impl<T> PartialEq for Matrix<T> where
T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        if self.M != other.M || self.N != other.N {
            return false
        } else {
            for i in 0..self.M {
                for j in 0..self.N {
                    if self[i][j] != other[i][j] {
                        return false
                    }
                }
            }
        }
        true
    }
}

impl<T> fmt::Display for Matrix<T> where
T: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fmt_str = String::new();
        for i in 0..self.M {
            fmt_str.push_str("\n| ");
            for j in 0..self.N {
                fmt_str.push_str(&format!("{} ", self[i][j]));
            }
            fmt_str.push_str("|");
        }
        write!(f, "{}", &fmt_str)
    }
}

impl<T: Mul<Output = T>> MatrixMatrixMul<T> for Matrix<T> where 
T: Default + Copy + AddAssign + std::fmt::Debug {
    fn mul(&self, other: &Self) -> Self {
        if self.N != other.M {
                panic!("Bad matrices dimensions.\nfirst.N != second.M\n");
        }
        let mut output = Matrix::new_empty(self.M, other.N);
        for i in 0..self.M {
            for j in 0..other.N {
                for k in 0..self.N {
                    output[i][j] += self[i][k] * other[k][j];  
                }
            }
        }
        output
    }
}

impl<T> Dot<T> for Vec<T> where
T: Mul<Output = T> + Copy + AddAssign + Default
{
    fn dot(&self, other: &Vec<T>) -> T {
        let mut dot: T = Default::default();
        for ind in 0..self.len() {
            dot += self[ind] * other[ind];
        }
        dot
    }
}

impl<T: Mul<Output = T>> VectorScalarMul<T> for Vec<T> where 
T: Copy + Default + Mul
{
    fn vsmul(&self, other: T) -> Self {
        let mut output = vec![Default::default();self.len()];
        for ind in 0..self.len() {
            output[ind] = self[ind] * other;
        }
        output
    }
}

impl<T: Add<Output = T>> VectorScalarAdd<T> for Vec<T> where 
T: Copy + Default + Add
{
    fn vsadd(&self, other: T) -> Self {
        let mut output = vec![Default::default();self.len()];
        for ind in 0..self.len() {
            output[ind] = self[ind] + other;
        }
        output
    }
}

impl<T: Add<Output = T>> ScalarVectorAdd<T> for T where 
T: Copy + Default + Add
{
    fn vsadd(&self, other: &Vec<T>) -> Vec<T> {
        let mut output = vec![Default::default();other.len()];
        for ind in 0..other.len() {
            output[ind] = *self + other[ind];
        }
        output
    }
}

impl<T: Mul<Output = T>> ScalarVectorMul<T> for T where
T: Copy + Default + Mul 
{
    fn vsmul(&self, other: &Vec<T>) -> Vec<T> {
        let mut output = vec![Default::default();other.len()];
        for ind in 0..other.len() {
            output[ind] = *self * other[ind];
        }
        output
    }
} 

impl<T: Add<Output = T>> VectorVectorAdd<T> for Vec<T> where 
T: Copy + Default + Add {
    fn vvadd(&self, other: &Self) -> Self {
        let mut output = vec![Default::default();self.len()];

        for ind in 0..self.len() {
            output[ind] = self[ind] + other[ind];
        }
        output
    }
}

impl<T> Output<T> for Perceptron<T> {
    fn output(&self) -> T {
        (self.integration_fn)(&self.inputs, &self.weights, &self.bias)
    }
}

#[allow(dead_code, non_snake_case)]
impl<T> Matrix<T> where
T: Default + Copy
{
    fn new(input_data: Vec<Vec<T>>) -> Matrix<T> {
        let M = input_data.len();
        let N = input_data[0].len();
        let mut data = vec![Default::default(); M*N];
        for i in 0..M {
            for j in 0..N {
                data[i * N + j] = input_data[i][j];
            }
        }

        Matrix { M, N, data }
    }

    fn new_identity(size: usize, val: T) -> Matrix<T> {
        let mut m = Matrix::new_empty(size, size);
        for i in 0..size {
            m[i][i] = val;
        }
        m
    }

    fn new_empty(M: usize, N: usize) -> Matrix<T> {
        let data = vec![Default::default(); M * N];
        Matrix { M, N, data }
    }

    fn from_raw(M: usize, N: usize, data: Vec<T>) -> Matrix<T> {
        if M * N != data.len() {
            panic!("Provided dimensions or data is incorrect\nM * N does not equal data.len()");
        }
        Matrix { M, N, data }
    }

    fn add_row(&mut self, row: &mut Vec<T>) {
        if self.N != row.len() {
            if self.data.len() != 0 {
                panic!("\n***\nBad dimensions\nself.N: {} != row.len(): {}\n***\n", self.N, row.len());
            } else {
                self.N = row.len();
            }
        }
        self.data.append(row);
        self.M += 1;
        
    }
}

impl<T> Matrix<T> where 
T: Copy + Mul + MulAssign + Add + AddAssign + Neg + Default {
    fn get_minors(&self) -> Vec<Minor<T>> {
        let mut minors_vec = Vec::with_capacity(self.M);
        for idx in 0..self.M {
            let mut data = Vec::new();
            data.reserve((self.M-1) * (self.M-1));
            for j in 1..self.N {
                data.push_multiple(&self[j][0..idx]);
                data.push_multiple(&self[j][idx+1..self.N]);
            }
            let matrix = Matrix::from_raw(self.M-1, self.N-1, data);
            let coef = self[0][idx];
            let minor = Minor{ coef, matrix };
            minors_vec.push(minor);
        }
        minors_vec
    }
}

impl<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T>> MinorDeterminant<T> for Minor<T> where
T: Default + Copy + Add + AddAssign + Mul + MulAssign + Sub + SubAssign + Neg {
    fn det(m: &Minor<T>) -> T {
        m.coef * Matrix::det(&m.matrix)
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];
    fn index<'a>(&'a self, row_idx: usize) -> &'a Self::Output {
        &self.data[row_idx * self.N .. (row_idx + 1) * self.N]
    }
}


impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut<'a>(&'a mut self, row_idx: usize) -> &'a mut Self::Output {
        &mut self.data[row_idx * self.N .. (row_idx + 1) * self.N]
    }
}

pub fn test_determinant_2x2() {
    let expected;
    let actual;
    let matrix;

    matrix = mat![1, 2; 3, 4];
    expected = -2;
    actual = Matrix::det(&matrix);

    test_determinant_nxn(matrix, expected);
}

pub fn test_determinant_3x3() {
    let expected;
    let actual;
    let matrix;

    matrix = mat![1, 2, 3; 4, 5, 6; 7, 8, 9];
    expected = 0;
    actual = Matrix::det(&matrix);

    test_determinant_nxn(matrix, expected);
}

pub fn test_determinant_4x4() {
    let expected;
    let matrix;

    matrix = mat![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16];
    expected = 0;

    test_determinant_nxn(matrix, expected);
}


pub fn test_determinant_nxn<T: Mul<Output = T> + Sub<Output = T> + Add<Output = T>>(matrix: Matrix<T>, expected: T) -> bool
where T: Copy + Add + AddAssign + Mul + MulAssign + Sub + SubAssign + Neg + PartialEq + Default + fmt::Display {
    //#useless code... but cute
    let actual;
    actual = Matrix::det(&matrix);
    match actual == expected {
        true => { 
            println!("test_determinant_{}x{} passed",matrix.M, matrix.N); 
            true },
        false => { 
            println!("\ntest_determinant_{}x{} failed", matrix.M, matrix.N);
            println!("\nactual = {} != {} = expected\n", actual, expected);
            false },
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_index() {
        let m = mat![1, 2; 3, 4];
        let r1 = &m[0];
        let r2 = &m[1];
        assert_eq!(r1[0], 1);
        assert_eq!(r1[1], 2);
        assert_eq!(r2[0], 3);
        assert_eq!(r2[1], 4);
    }

    #[test]
    fn test_matrix_eq() {
        let m1 = mat![1, 2; 3, 4];
        let m2 = mat![1, 2; 3, 4];
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_matrix_mul() {
        let m1 = mat![-2, 3; 3, -4];
        let m2 = mat![4, 3; 3, 2];
        let expected = mat![1, 0; 0, 1];
        let result = m1.mul(&m2);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_matrix_new_identity() {
        let m = Matrix::new_identity(2, 1);
        let expected = mat![1, 0; 0, 1];
        assert_eq!(expected, m);
    }

    #[test]
    fn test_push_multiple() {
        let initial_v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut new_v = Vec::with_capacity(4);
        new_v.push_multiple(&initial_v[4..8]);
        assert_eq!(new_v.len(), 4);
        assert_eq!(new_v[0], 4);
        assert_eq!(new_v[3], 7);
    }

    #[test]
    fn test_determinant_2x2() {
        let matrix = mat![1, 2; 3, 4];
        let expected = -2;
        assert!(super::test_determinant_nxn(matrix, expected));
    }

    #[test]
    fn test_determinant_3x3() {
        let matrix = mat![1, 2, 3; 4, 5, 6; 7, 8, 9];
        let expected = 0;
        assert!(super::test_determinant_nxn(matrix, expected));
    }

    #[test]
    fn test_determinant_4x4() {
        let matrix = mat![1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12; 13, 14, 15, 16];
        let expected = 0;
        assert!(super::test_determinant_nxn(matrix, expected));
    } 

}

fn main() {
    ml_main();
}

#[allow(dead_code, unused_variables)]
fn ml_main() {
    let activation_fn = 
        |integration_result| 
            if integration_result >= 0 
                {1} 
            else 
                {0};

    let integration_fn = 
        |inputs: &Vec<i32>, weights: &Vec<i32>, b: &i32| 
            inputs.dot(weights) + b;
            
    let inputs = vec![1,-1,2];
    let weights = vec![1,5,2];
    let bias = 2;
    let dot = inputs.dot(&weights);
    println!("{:?} dot {:?} = {}", inputs, weights, dot);

    let perceptron = Perceptron {   
        inputs,
        weights,
        bias,
        integration_fn,
        activation_fn,
    };
    println!("perceptron output: {}", perceptron.output());

    let v1 = vec![4,3,-1];
    let v2 = vec![-1,1,2];
    let f  = 3;

    let result = v1.vvadd(&3.vsmul(&v2));

    println!("{:?} + {} * {:?} = {:?}",v1, f, v2, result);

    
    let a = 2;
    let b = 3;
    let c = a * b;

    println!("{} * {} = {}", a, b, c);


    let m1 = mat![-2, 3; 3, -4];
    println!("{}", m1);
    let m2 = mat![4, 3; 3, 2];
    println!("{}", m2);
    let result = m1.mul(&m2);
    println!("{}", result);

    println!("{} * {} = {}", m1, m2, result);

    test_determinant_2x2();

    let matrix = mat![1, 2, 3; 4, 5, 6; 7, 8, 9];
    let expected = 0;

    let result = test_determinant_nxn(matrix, expected);



}

mod question_photo {
    use super::*;
    pub fn question_photo_main() {
        println!("Question Photo Julie");
        let m = 1;
        let n = 14;
        let s = Summation::<nCk>::new(m, n);
        println!("{:#?}", s);
    }

    #[allow(non_camel_case_types)]
    #[derive(Debug)]
    struct nCk {
        n: u32,
        k: u32,
        result: u32
    }

    #[derive(Debug)]
    struct Summation<T> where
    T: Output<u32> + IterationResult<T>
    {
        m: u32,
        n: u32,
        values: Vec<T>,
        result: u32
    }

    impl nCk {
        fn new(n: u32, k: u32) -> nCk {
            nCk { n, k, result: binomial_coefficient(n, k) }
        }
    }

    impl Output<u32> for nCk {
        fn output(&self) -> u32 {
            self.result
        }
    }

    trait IterationResult<T> {
        fn iteration_result(i: u32, n: u32) -> T;
    }

    impl IterationResult<nCk> for nCk {
        fn iteration_result(i: u32, n: u32) -> nCk {
            nCk::new(n, i)
        }
    }

    impl<T> Summation<T> where
    T: Output<u32> + IterationResult<T>
    {
        fn new(m: u32, n: u32) -> Summation<T> {
            if n < m {
                panic!("n can't be inferior to n");
            }
            let mut values = Vec::new();
            values.reserve_exact((n - m + 1).try_into().unwrap());
            let mut result = 0;
            for i in m..n+1 {
                let iteration_value = T::iteration_result(i, n);
                result += iteration_value.output();
                values.push(iteration_value);
            }
            Summation{ m, n, values, result }
        }
    }

    fn binomial_coefficient(n: u32, k: u32) -> u32 {
        if n < k {
            panic!("\nn must be greater or equal to k!\n");
        }
        let nmk = n - k;
        let pair = match nmk < k {
            true  => (nmk, k),
            false => (k, nmk)};
        let range_numerator = pair.1 + 1..n+1;
        let range_denominator = 1..pair.0 + 1;
        let mut numerator = 1;
        for v in range_numerator {
            numerator *= v;
        }
        let mut denominator = 1;
        for v in range_denominator {
            denominator *= v;
        }
        numerator / denominator
    }
}